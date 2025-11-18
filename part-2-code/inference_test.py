#!/usr/bin/env python3
"""
Generate test predictions (both .sql and .pkl) using trained model
"""

import os
import argparse
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from load_data import get_dataloader
from utils import save_queries_and_records

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Generate test predictions')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., checkpoints/t5_ft/exp_name/best_model.pt)')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Experiment name for output files')
    parser.add_argument('--use_schema', action='store_true', default=True,
                       help='Use schema context (should match training)')
    parser.add_argument('--use_preprocessed', action='store_true', default=True,
                       help='Use preprocessed data (should match training)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--max_gen_length', type=int, default=512,
                       help='Max generation length')
    parser.add_argument('--num_beams', type=int, default=1,
                       help='Number of beams for beam search')
    
    args = parser.parse_args()
    return args

def test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path):
    """Generate SQL predictions and compute records"""
    model.eval()
    sql_queries = []
    
    print(f"\nGenerating SQL predictions for test set...")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num beams: {args.num_beams}")
    print("="*80)
    
    progress_bar = tqdm(test_loader, desc="Generating")
    
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in progress_bar:
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate
            if args.num_beams > 1:
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.max_gen_length,
                    num_beams=args.num_beams,
                    early_stopping=True,
                )
            else:
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=args.max_gen_length,
                )
            
            # Decode
            for gen_ids in generated_ids:
                sql = tokenizer.decode(gen_ids, skip_special_tokens=True)
                sql_queries.append(sql)
    
    print(f"\n✓ Generated {len(sql_queries)} SQL queries")
    
    # Save queries and compute records
    print(f"\nSaving queries to: {model_sql_path}")
    print(f"Computing and saving records to: {model_record_path}")
    print("(This may take a few minutes to execute all SQL queries...)")
    
    save_queries_and_records(sql_queries, model_sql_path, model_record_path)
    
    print(f"\n{'='*80}")
    print(f"✓ SUCCESS!")
    print(f"{'='*80}")
    print(f"Files created:")
    print(f"  1. {model_sql_path}")
    print(f"  2. {model_record_path}")
    print(f"\nYou can now submit these files to Gradescope!")
    print(f"{'='*80}\n")
    
    # Show sample predictions
    print("\nSample predictions (first 5):")
    print("-"*80)
    nl_path = 'data/test.nl'
    with open(nl_path, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    for i in range(min(5, len(sql_queries))):
        print(f"\n{i+1}. NL: {nl_queries[i]}")
        print(f"   SQL: {sql_queries[i]}")
    print("-"*80 + "\n")
    
    return sql_queries

def main():
    args = get_args()
    
    print("="*80)
    print("TEST SET INFERENCE")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Schema: {args.use_schema}")
    print(f"Preprocessed: {args.use_preprocessed}")
    print("="*80 + "\n")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                print(f"  - {os.path.join(checkpoint_dir, f)}")
        return
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Load test data
    print(f"Loading test data (schema={args.use_schema}, preprocessed={args.use_preprocessed})...")
    test_loader = get_dataloader(
        args.batch_size, 
        "test", 
        use_schema=args.use_schema,
        use_preprocessed=args.use_preprocessed
    )
    print(f"✓ Loaded {len(test_loader)} batches\n")
    
    # Load model
    print("Initializing model architecture...")
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    
    print(f"Loading checkpoint weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ Model loaded on {DEVICE}\n")
    
    # Set output paths
    model_sql_path = f'results/t5_ft_{args.experiment_name}_test.sql'
    model_record_path = f'results/t5_ft_{args.experiment_name}_test.pkl'
    
    # Generate predictions
    test_inference(args, model, test_loader, tokenizer, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()