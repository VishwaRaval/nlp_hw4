"""
Generate ground truth records for dev and test sets
"""
import os
import pickle
from utils import read_queries, compute_records

def generate_ground_truth_records():
    """Generate ground truth records for evaluation"""
    
    print("=" * 80)
    print("GENERATING GROUND TRUTH RECORDS")
    print("=" * 80)
    
    # Create records directory if it doesn't exist
    os.makedirs("records", exist_ok=True)
    
    # Generate for dev set
    print("\n[1/2] Generating dev set records...")
    dev_sql_path = "data/dev.sql"  # Use original data
    dev_output_path = "records/ground_truth_dev.pkl"
    
    if not os.path.exists(dev_sql_path):
        print(f"✗ ERROR: {dev_sql_path} not found!")
        return False
    
    dev_queries = read_queries(dev_sql_path)
    print(f"  Found {len(dev_queries)} queries")
    
    print(f"  Computing records (this may take 1-2 minutes)...")
    dev_records, dev_errors = compute_records(dev_queries)
    
    with open(dev_output_path, 'wb') as f:
        pickle.dump((dev_records, dev_errors), f)
    
    print(f"✓ Saved to {dev_output_path}")
    
    # Count errors
    num_errors = sum(1 for err in dev_errors if err)
    print(f"  {len(dev_queries)} queries, {num_errors} had errors")
    
    # Generate for test set (optional, no ground truth SQL)
    print("\n[2/2] Generating test set records...")
    test_sql_path = "data/test.sql"
    test_output_path = "records/ground_truth_test.pkl"
    
    if os.path.exists(test_sql_path):
        test_queries = read_queries(test_sql_path)
        print(f"  Found {len(test_queries)} queries")
        
        print(f"  Computing records (this may take 1-2 minutes)...")
        test_records, test_errors = compute_records(test_queries)
        
        with open(test_output_path, 'wb') as f:
            pickle.dump((test_records, test_errors), f)
        
        print(f"✓ Saved to {test_output_path}")
        
        num_errors = sum(1 for err in test_errors if err)
        print(f"  {len(test_queries)} queries, {num_errors} had errors")
    else:
        print(f"⚠️  {test_sql_path} not found (test set has no ground truth SQL)")
        print(f"  Skipping test ground truth generation")
    
    print("\n" + "=" * 80)
    print("✓ Ground truth records generated successfully!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = generate_ground_truth_records()
    exit(0 if success else 1)