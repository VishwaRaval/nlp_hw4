#!/usr/bin/env python3
"""
Evaluate all dev set predictions using Record F1 score (the actual grading metric)
"""

import os
import glob
import pickle
from utils import compute_metrics

def main():
    # Paths
    results_dir = "results"
    records_dir = "records"
    gold_sql = "data/dev.sql"
    gold_records = "records/ground_truth_dev.pkl"
    
    # Check if ground truth exists
    if not os.path.exists(gold_records):
        print(f"Ground truth records not found. Creating {gold_records}...")
        from utils import save_queries_and_records
        with open(gold_sql, 'r') as f:
            gt_queries = [line.strip() for line in f.readlines()]
        save_queries_and_records(gt_queries, gold_sql, gold_records)
        print(f"✓ Created ground truth records\n")
    
    # Find all dev prediction files (both SQL and PKL)
    dev_sql_files = glob.glob(f"{results_dir}/*_dev_*.sql")
    dev_pkl_files = glob.glob(f"{records_dir}/*_dev_*.pkl")
    
    # Match SQL files with their corresponding PKL files
    results = []
    
    print("="*100)
    print("EVALUATING ALL DEV PREDICTIONS")
    print("="*100)
    print(f"Found {len(dev_sql_files)} SQL files and {len(dev_pkl_files)} PKL files\n")
    
    for sql_file in sorted(dev_sql_files):
        # Construct corresponding PKL file path
        basename = os.path.basename(sql_file).replace('.sql', '.pkl')
        pkl_file = os.path.join(records_dir, basename)
        
        if not os.path.exists(pkl_file):
            print(f"⚠ Skipping {basename} - no corresponding .pkl file")
            continue
        
        try:
            # Compute metrics
            sql_em, record_em, record_f1, error_msgs = compute_metrics(
                gold_sql, sql_file, gold_records, pkl_file
            )
            
            # Extract experiment name and epoch
            filename = os.path.basename(sql_file)
            
            # Parse experiment name
            parts = filename.replace('t5_ft_', '').replace('.sql', '').split('_dev_')
            exp_name = parts[0] if len(parts) > 0 else filename
            
            # Parse epoch
            epoch = None
            if len(parts) > 1:
                epoch_part = parts[1].replace('epoch', '')
                try:
                    epoch = int(epoch_part)
                except ValueError:
                    epoch = epoch_part
            
            # Count errors
            with open(pkl_file, 'rb') as f:
                records, error_list = pickle.load(f)
            num_errors = sum(1 for e in error_list if e)
            error_rate = num_errors / len(error_list) * 100 if error_list else 0
            
            results.append({
                'exp_name': exp_name,
                'epoch': epoch,
                'filename': filename,
                'sql_em': sql_em,
                'record_em': record_em,
                'record_f1': record_f1,
                'num_errors': num_errors,
                'error_rate': error_rate,
            })
            
            print(f"{filename:70s} | F1: {record_f1:.4f} | EM: {record_em:.4f} | Errors: {num_errors:3d} ({error_rate:5.1f}%)")
            
        except Exception as e:
            print(f"✗ Error evaluating {sql_file}: {e}")
            continue
    
    if not results:
        print("\n❌ No valid results found!")
        return
    
    # Sort by F1 score
    results_sorted = sorted(results, key=lambda x: x['record_f1'], reverse=True)
    
    # Print summary
    print("\n" + "="*100)
    print("TOP 10 MODELS BY RECORD F1 SCORE")
    print("="*100)
    print(f"{'Rank':<5} {'Experiment':<35} {'Epoch':<7} {'F1':<8} {'EM':<8} {'SQL EM':<8} {'Errors':<10}")
    print("-"*100)
    
    for i, result in enumerate(results_sorted[:10], 1):
        epoch_str = f"{result['epoch']}" if result['epoch'] is not None else "N/A"
        print(f"{i:<5} {result['exp_name']:<35} {epoch_str:<7} "
              f"{result['record_f1']:.4f}   {result['record_em']:.4f}   "
              f"{result['sql_em']:.4f}   {result['num_errors']:3d} ({result['error_rate']:4.1f}%)")
    
    # Group by experiment and show best epoch for each
    from collections import defaultdict
    by_experiment = defaultdict(list)
    for result in results:
        by_experiment[result['exp_name']].append(result)
    
    print("\n" + "="*100)
    print("BEST EPOCH FOR EACH EXPERIMENT")
    print("="*100)
    print(f"{'Experiment':<35} {'Best Epoch':<12} {'F1':<8} {'EM':<8} {'Errors':<10}")
    print("-"*100)
    
    exp_best = []
    for exp_name, exp_results in sorted(by_experiment.items()):
        best = max(exp_results, key=lambda x: x['record_f1'])
        epoch_str = f"{best['epoch']}" if best['epoch'] is not None else "N/A"
        print(f"{exp_name:<35} {epoch_str:<12} {best['record_f1']:.4f}   "
              f"{best['record_em']:.4f}   {best['num_errors']:3d} ({best['error_rate']:4.1f}%)")
        exp_best.append(best)
    
    # Overall best
    best_overall = max(results, key=lambda x: x['record_f1'])
    
    print("\n" + "="*100)
    print("🏆 BEST OVERALL MODEL")
    print("="*100)
    print(f"Experiment: {best_overall['exp_name']}")
    print(f"Epoch: {best_overall['epoch']}")
    print(f"Filename: {best_overall['filename']}")
    print(f"Record F1: {best_overall['record_f1']:.4f}")
    print(f"Record EM: {best_overall['record_em']:.4f}")
    print(f"SQL EM: {best_overall['sql_em']:.4f}")
    print(f"Errors: {best_overall['num_errors']} ({best_overall['error_rate']:.1f}%)")
    
    # Determine checkpoint path
    checkpoint_path = f"checkpoints/t5_ft/{best_overall['exp_name']}/best_model.pt"
    
    print("\n" + "="*100)
    print("📋 NEXT STEPS")
    print("="*100)
    print(f"1. Generate test predictions using the best model:")
    print(f"\n   python inference_test.py \\")
    print(f"       --checkpoint {checkpoint_path} \\")
    print(f"       --experiment_name {best_overall['exp_name']} \\")
    print(f"       --use_schema \\")
    print(f"       --use_preprocessed \\")
    print(f"       --batch_size 16")
    print(f"\n2. Submit these files to Gradescope:")
    print(f"   - results/t5_ft_{best_overall['exp_name']}_test.sql")
    print(f"   - results/t5_ft_{best_overall['exp_name']}_test.pkl")
    print("="*100 + "\n")
    
    # Show top 3 for submission consideration
    print("🎯 TOP 3 MODELS TO CONSIDER FOR SUBMISSION:")
    print("-"*100)
    for i, result in enumerate(results_sorted[:3], 1):
        print(f"\n{i}. {result['exp_name']} (epoch {result['epoch']})")
        print(f"   F1: {result['record_f1']:.4f}, EM: {result['record_em']:.4f}, Errors: {result['num_errors']}")
        print(f"   Checkpoint: checkpoints/t5_ft/{result['exp_name']}/best_model.pt")

if __name__ == "__main__":
    main()