import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")


def get_args():
    """
    Arguments for training.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=20)
    parser.add_argument("--patience_epochs", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--use_schema", action="store_true")
    parser.add_argument("--use_preprocessed", action="store_true")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--run_error_analysis", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    evals_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    experiment_name = args.experiment_name
    data_dir = "data_preprocessed" if args.use_preprocessed else "data"
    gt_sql_path = os.path.join(data_dir, "dev.sql")
    gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
    model_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_dev.sql")
    model_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_dev.pkl")
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}")

        if args.use_wandb:
            wandb.log({"train/loss": tr_loss, "epoch": epoch}, step=epoch)

        is_last_epoch = (epoch == args.max_n_epochs - 1)
        should_eval = (epoch + 1) % args.eval_every == 0 or is_last_epoch

        if not should_eval:
            continue

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )

        print(
            f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, "
            f"Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}"
        )
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of generated outputs had SQL errors")

        if args.use_wandb:
            wandb.log({
                "dev/loss": eval_loss,
                "dev/record_f1": record_f1,
                "dev/record_em": record_em,
                "dev/sql_em": sql_em,
                "dev/error_rate": error_rate,
                "epoch": epoch,
            }, step=epoch)

        if record_f1 > best_f1:
            print(f"✓ New best F1: {record_f1:.4f} (previous: {best_f1:.4f})")
            best_f1 = record_f1
            evals_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
        else:
            evals_since_improvement += 1
            print(f"No improvement for {evals_since_improvement} eval(s)")

        save_model(checkpoint_dir, model, best=False)

        if evals_since_improvement >= args.patience_epochs:
            print(f"Early stopping after {evals_since_improvement} evals without improvement")
            break
    
    print(f"\nTraining completed. Best F1: {best_f1:.4f}")


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for encoder_input, encoder_mask, decoder_targets, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Pass labels directly - T5 handles the shifting
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=decoder_targets,
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            non_pad = (decoder_targets != PAD_IDX)
            num_tokens = non_pad.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_generated_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_targets, initial_decoder_inputs in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Loss with teacher forcing
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=decoder_targets,
            )
            loss = outputs.loss

            non_pad = (decoder_targets != PAD_IDX)
            num_tokens = non_pad.sum().item()
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

            # Generate predictions
            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            gen_sql = TOKENIZER.batch_decode(gen_ids, skip_special_tokens=True)
            all_generated_sql.extend(gen_sql)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    num_err = sum(1 for msg in model_error_msgs if msg)
    error_rate = num_err / len(model_error_msgs) if model_error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    all_generated_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader, desc="Test inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            gen_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            gen_sql = TOKENIZER.batch_decode(gen_ids, skip_special_tokens=True)
            all_generated_sql.extend(gen_sql)

    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)
    print(f"Saved test SQL to {model_sql_path} and records to {model_record_path}")


def main():
    args = get_args()
    
    print("=" * 80)
    print("T5 TEXT-TO-SQL TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Fine-tune: {args.finetune}")
    print(f"LR: {args.learning_rate}, Batch: {args.batch_size}, Beams: {args.num_beams}")
    print(f"Schema: {args.use_schema}, Preprocessed: {args.use_preprocessed}")
    print("=" * 80)
    
    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size,
        use_schema=args.use_schema, use_preprocessed=args.use_preprocessed,
    )
    
    print(f"Loaded: {len(train_loader)} train, {len(dev_loader)} dev, {len(test_loader)} test batches")
    
    model = initialize_model(args)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    model = load_model_from_checkpoint(args, best=True)

    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"
    data_dir = "data_preprocessed" if args.use_preprocessed else "data"
    
    gt_sql_path = os.path.join(data_dir, "dev.sql")
    gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
    model_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_dev.sql")
    model_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_dev.pkl")
    
    dev_loss, dev_f1, dev_em, dev_sql_em, dev_err = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    print(f"DEV: F1={dev_f1:.4f}, EM={dev_em:.4f}, SQL_EM={dev_sql_em:.4f}, Err={dev_err*100:.2f}%")

    model_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_test.sql")
    model_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print("✓ Test predictions saved!")


if __name__ == "__main__":
    main()