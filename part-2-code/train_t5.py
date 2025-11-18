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
    Arguments for training. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument(
        "--finetune", action="store_true", help="Whether to finetune T5 or not"
    )

    # Training hyperparameters
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="AdamW",
        choices=["AdamW"],
        help="What optimizer to use",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="Whether to use a LR scheduler and what type to use if so",
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=0,
        help="How many epochs to warm up the learning rate for if using a scheduler",
    )
    parser.add_argument(
        "--max_n_epochs",
        type=int,
        default=0,
        help="How many epochs to train the model for",
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=0,
        help="If validation performance stops improving, how many evals should we wait before stopping?",
    )

    # New: how often to run dev evaluation
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Run dev evaluation every N epochs",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, we will use wandb to keep track of experiments",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="How should we name this experiment?",
    )

    # Data / model behavior hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    # Extra flags you mentioned (wired into CLI so your command works)
    parser.add_argument(
        "--use_schema",
        action="store_true",
        help="Placeholder flag for including schema; handled in data loading if used",
    )
    parser.add_argument(
        "--use_preprocessed",
        action="store_true",
        help="Placeholder flag for using preprocessed NL; handled in data loading if used",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use in T5.generate",
    )
    parser.add_argument(
        "--run_error_analysis",
        action="store_true",
        help="Placeholder flag to optionally trigger extra error analysis",
    )

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    evals_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join(
        "checkpoints", f"{model_type}_experiments", args.experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    experiment_name = "ft_experiment"
    data_dir = "data_preprocessed" if args.use_preprocessed else "data"
    gt_sql_path = os.path.join(data_dir, "dev.sql")
    gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
    model_sql_path = os.path.join(
        "results", f"t5_{model_type}_{experiment_name}_dev.sql"
    )
    model_record_path = os.path.join(
        "records", f"t5_{model_type}_{experiment_name}_dev.pkl"
    )

    for epoch in range(args.max_n_epochs):
        # ----------------- TRAIN ONE EPOCH -----------------
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        if args.use_wandb:
            wandb.log({"train/loss": tr_loss}, step=epoch)

        # ----------------- DECIDE WHETHER TO EVALUATE -----------------
        is_last_epoch = (epoch == args.max_n_epochs - 1)
        should_eval = (epoch + 1) % args.eval_every == 0 or is_last_epoch

        if not should_eval:
            continue  # skip dev eval this epoch

        # ----------------- DEV EVAL -----------------
        (
            eval_loss,
            record_f1,
            record_em,
            sql_em,
            error_rate,
        ) = eval_epoch(
            args,
            model,
            dev_loader,
            gt_sql_path,
            model_sql_path,
            gt_record_path,
            model_record_path,
        )

        print(
            f"Epoch {epoch}: Dev loss: {eval_loss}, "
            f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}"
        )
        print(
            f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors"
        )

        if args.use_wandb:
            result_dict = {
                "train/loss": tr_loss,
                "dev/loss": eval_loss,
                "dev/record_f1": record_f1,
                "dev/record_em": record_em,
                "dev/sql_em": sql_em,
                "dev/error_rate": error_rate,
            }
            wandb.log(result_dict, step=epoch)

        # ----------------- EARLY STOPPING + CHECKPOINTS -----------------
        if record_f1 > best_f1:
            best_f1 = record_f1
            evals_since_improvement = 0
            # Save best model whenever we improve
            save_model(checkpoint_dir, model, best=True)
        else:
            evals_since_improvement += 1

        # Always save "last" model on eval epochs
        save_model(checkpoint_dir, model, best=False)

        if evals_since_improvement >= args.patience_epochs:
            print(
                f"Early stopping triggered after {evals_since_improvement} evals "
                f"without improvement (patience={args.patience_epochs})."
            )
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(
        train_loader
    ):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )["logits"]

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(
    args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path
):
    """
    Evaluation loop on dev set.

    Returns:
        avg_loss, record_f1, record_em, sql_em, error_rate
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0

    all_generated_sql = []

    with torch.no_grad():
        for (
            encoder_input,
            encoder_mask,
            decoder_input,
            decoder_targets,
            initial_decoder_inputs,
        ) in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Teacher-forcing loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )
            logits = outputs["logits"]  # (B, T, V)

            non_pad = decoder_targets != PAD_IDX
            if non_pad.sum().item() == 0:
                continue
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            num_tokens = non_pad.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generation for metrics
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

    # Save generated SQL + records, then compute metrics
    save_queries_and_records(all_generated_sql, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    # error rate = fraction of queries that produced a non-empty error message
    num_err = sum(1 for msg in model_error_msgs if msg)
    error_rate = num_err / len(model_error_msgs) if model_error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Generate SQL queries for the test set and save queries + records.
    """
    model.eval()
    all_generated_sql = []

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
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
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size,
        args.test_batch_size,
        use_schema=args.use_schema,
        use_preprocessed=args.use_preprocessed,
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate best checkpoint on dev + run test inference
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name = "ft_experiment"
    model_type = "ft" if args.finetune else "scr"

    # Dev set
    data_dir = "data_preprocessed" if args.use_preprocessed else "data"
    gt_sql_path = os.path.join(data_dir, "dev.sql")
    gt_record_path = os.path.join("records", "ground_truth_dev.pkl")
    model_sql_path = os.path.join(
        "results", f"t5_{model_type}_{experiment_name}_dev.sql"
    )
    model_record_path = os.path.join(
        "records", f"t5_{model_type}_{experiment_name}_dev.pkl"
    )
    (
        dev_loss,
        dev_record_f1,
        dev_record_em,
        dev_sql_em,
        dev_error_rate,
    ) = eval_epoch(
        args,
        model,
        dev_loader,
        gt_sql_path,
        model_sql_path,
        gt_record_path,
        model_record_path,
    )
    print(
        f"Dev set results: Loss: {dev_loss}, "
        f"Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}"
    )
    print(
        f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors"
    )

    # Test set
    model_sql_path = os.path.join(
        "results", f"t5_{model_type}_{experiment_name}_test.sql"
    )
    model_record_path = os.path.join(
        "records", f"t5_{model_type}_{experiment_name}_test.pkl"
    )
    test_inference(args, model, test_loader, model_sql_path, model_record_path)


if __name__ == "__main__":
    main()
