import os
import argparse
from collections import defaultdict

def load_queries(path):
    """Load SQL queries from file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_nl_queries(path):
    """Load natural language queries."""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_errors(gt_path, pred_path, nl_path, output_path):
    """
    Analyze prediction errors and categorize them.
    
    Args:
        gt_path: Path to ground truth SQL
        pred_path: Path to predicted SQL
        nl_path: Path to natural language queries
        output_path: Path to save analysis
    """
    gt_queries = load_queries(gt_path)
    pred_queries = load_queries(pred_path)
    nl_queries = load_nl_queries(nl_path)
    
    assert len(gt_queries) == len(pred_queries) == len(nl_queries)
    
    # Error categories
    errors = defaultdict(list)
    
    print(f"\nAnalyzing {len(gt_queries)} queries...")
    
    total_errors = 0
    for idx, (nl, gt, pred) in enumerate(zip(nl_queries, gt_queries, pred_queries)):
        if gt.strip() == pred.strip():
            continue  # Correct
        
        total_errors += 1
        error_type = categorize_error(nl, gt, pred)
        errors[error_type].append({
            'idx': idx,
            'nl': nl,
            'gt': gt,
            'pred': pred
        })
    
    # Print summary
    print(f"\nTotal errors: {total_errors}/{len(gt_queries)} ({100*total_errors/len(gt_queries):.2f}%)")
    print(f"Accuracy: {100*(1-total_errors/len(gt_queries)):.2f}%")
    print("\nError breakdown:")
    for error_type, examples in sorted(errors.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {error_type}: {len(examples)} ({100*len(examples)/total_errors:.1f}%)")
    
    # Save detailed analysis
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total queries: {len(gt_queries)}\n")
        f.write(f"Total errors: {total_errors}\n")
        f.write(f"Accuracy: {100*(1-total_errors/len(gt_queries)):.2f}%\n\n")
        
        f.write("ERROR CATEGORIES:\n")
        f.write("-"*80 + "\n")
        
        for error_type, examples in sorted(errors.items(), key=lambda x: len(x[1]), reverse=True):
            f.write(f"\n{error_type}: {len(examples)}/{total_errors} ({100*len(examples)/total_errors:.1f}%)\n")
            f.write("="*80 + "\n\n")
            
            # Show first 3 examples of each type
            for i, ex in enumerate(examples[:3]):
                f.write(f"Example {i+1} (Index: {ex['idx']}):\n")
                f.write(f"NL: {ex['nl']}\n\n")
                f.write(f"Ground Truth:\n{ex['gt']}\n\n")
                f.write(f"Predicted:\n{ex['pred']}\n\n")
                f.write("-"*80 + "\n\n")
    
    print(f"\n✓ Detailed analysis saved to: {output_path}")
    
    # Print examples for Table 5
    print("\n" + "="*80)
    print("EXAMPLES FOR TABLE 5 (Report)")
    print("="*80)
    
    for error_type, examples in sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)[:3]:
        ex = examples[0]  # First example
        print(f"\nError Type: {error_type}")
        print(f"Statistics: {len(examples)}/{len(gt_queries)}")
        print(f"Example:")
        print(f"  NL: {ex['nl']}")
        print(f"  Predicted: {ex['pred'][:100]}...")
        print(f"  Gold: {ex['gt'][:100]}...")
        print()

def categorize_error(nl, gt, pred):
    """Categorize error type based on SQL differences."""
    gt_lower = gt.lower()
    pred_lower = pred.lower()
    
    # Missing or wrong JOIN
    if 'join' in gt_lower and 'join' not in pred_lower:
        return "Missing JOIN"
    if pred_lower.count('join') != gt_lower.count('join'):
        return "Wrong number of JOINs"
    
    # Aggregation errors
    agg_funcs = ['min', 'max', 'sum', 'avg', 'count']
    gt_aggs = [f for f in agg_funcs if f in gt_lower]
    pred_aggs = [f for f in agg_funcs if f in pred_lower]
    if set(gt_aggs) != set(pred_aggs):
        return "Wrong aggregation function"
    
    # GROUP BY errors
    if ('group by' in gt_lower) != ('group by' in pred_lower):
        return "Missing/Extra GROUP BY"
    
    # HAVING errors
    if ('having' in gt_lower) != ('having' in pred_lower):
        return "Missing/Extra HAVING"
    
    # ORDER BY errors
    if ('order by' in gt_lower) != ('order by' in pred_lower):
        return "Missing/Extra ORDER BY"
    
    # LIMIT errors
    if ('limit' in gt_lower) != ('limit' in pred_lower):
        return "Missing/Extra LIMIT"
    
    # Subquery errors
    gt_subqueries = gt_lower.count('select')
    pred_subqueries = pred_lower.count('select')
    if gt_subqueries != pred_subqueries:
        return "Wrong subquery structure"
    
    # WHERE clause errors
    if ('where' in gt_lower) != ('where' in pred_lower):
        return "Missing/Extra WHERE clause"
    
    # Column/table name errors
    if len(pred) < len(gt) * 0.7 or len(pred) > len(gt) * 1.5:
        return "Significantly different query length"
    
    return "Other/Complex error"

def main():
    parser = argparse.ArgumentParser(description='Error analysis for T5 predictions')
    parser.add_argument('--gt_sql', type=str, default='data/dev.sql',
                       help='Ground truth SQL queries')
    parser.add_argument('--pred_sql', type=str, required=True,
                       help='Predicted SQL queries')
    parser.add_argument('--nl_queries', type=str, default='data/dev.nl',
                       help='Natural language queries')
    parser.add_argument('--output', type=str, default='error_analysis.txt',
                       help='Output file for analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pred_sql):
        print(f"Error: Prediction file not found: {args.pred_sql}")
        print("\nAvailable prediction files:")
        if os.path.exists('results'):
            for f in os.listdir('results'):
                if 'dev' in f and f.endswith('.sql'):
                    print(f"  results/{f}")
        return
    
    analyze_errors(args.gt_sql, args.pred_sql, args.nl_queries, args.output)

if __name__ == "__main__":
    main()