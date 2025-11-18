#!/usr/bin/env python3
"""
Pre-training validation script to catch bugs before wasting GPU time
"""
import os
import sys

def check_file_exists(path, critical=True):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"✓ {path}")
        return True
    else:
        symbol = "✗" if critical else "⚠️"
        print(f"{symbol} MISSING: {path}")
        return not critical

def check_data_files():
    """Check all required data files exist"""
    print("\n" + "="*80)
    print("CHECKING DATA FILES")
    print("="*80)
    
    all_ok = True
    
    # Preprocessed data (since you're using --use_preprocessed)
    for split in ['train', 'dev']:
        all_ok &= check_file_exists(f"data_preprocessed/{split}.nl")
        all_ok &= check_file_exists(f"data_preprocessed/{split}.sql")
    
    all_ok &= check_file_exists("data_preprocessed/test.nl")
    
    # Schema
    all_ok &= check_file_exists("data/flight_database.schema")
    all_ok &= check_file_exists("data/flight_database.db")
    
    # Ground truth records
    all_ok &= check_file_exists("records/ground_truth_dev.pkl")
    check_file_exists("records/ground_truth_test.pkl", critical=False)
    
    return all_ok

def validate_data_loader():
    """Test that data loader works and returns correct format"""
    print("\n" + "="*80)
    print("VALIDATING DATA LOADER")
    print("="*80)
    
    try:
        from load_data import load_t5_data
        
        # Test loading with your settings
        print("Loading data with use_schema=True, use_preprocessed=True...")
        train_loader, dev_loader, test_loader = load_t5_data(
            batch_size=4,
            test_batch_size=4,
            use_schema=True,
            use_preprocessed=True
        )
        
        print(f"✓ Data loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Dev batches: {len(dev_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Get one batch from train
        train_batch = next(iter(train_loader))
        print(f"\n✓ Train batch structure:")
        print(f"  Number of elements: {len(train_batch)} (expected: 4)")
        print(f"  Shapes: {[t.shape if hasattr(t, 'shape') else type(t) for t in train_batch]}")
        
        if len(train_batch) != 4:
            print("  ✗ ERROR: Expected 4 elements!")
            return False
        
        encoder_input, encoder_mask, decoder_targets, initial_decoder_inputs = train_batch
        print(f"  encoder_input: {encoder_input.shape}")
        print(f"  encoder_mask: {encoder_mask.shape}")
        print(f"  decoder_targets: {decoder_targets.shape}")
        print(f"  initial_decoder_inputs: {initial_decoder_inputs.shape}")
        
        # Get one batch from dev
        dev_batch = next(iter(dev_loader))
        print(f"\n✓ Dev batch structure:")
        print(f"  Number of elements: {len(dev_batch)} (expected: 4)")
        
        # Get one batch from test
        test_batch = next(iter(test_loader))
        print(f"\n✓ Test batch structure:")
        print(f"  Number of elements: {len(test_batch)} (expected: 3)")
        
        if len(test_batch) != 3:
            print("  ✗ ERROR: Expected 3 elements for test!")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_model_loading():
    """Test that model can be initialized"""
    print("\n" + "="*80)
    print("CHECKING MODEL INITIALIZATION")
    print("="*80)
    
    try:
        from transformers import T5ForConditionalGeneration
        import torch
        
        print("Loading T5-small...")
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy data
        print("\nTesting forward pass...")
        dummy_input = torch.randint(0, 100, (2, 10))
        dummy_labels = torch.randint(0, 100, (2, 15))
        
        outputs = model(input_ids=dummy_input, labels=dummy_labels)
        print(f"✓ Forward pass successful")
        print(f"  Loss: {outputs.loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop_format():
    """Verify train_epoch will work with data format"""
    print("\n" + "="*80)
    print("TESTING TRAINING LOOP FORMAT")
    print("="*80)
    
    try:
        from load_data import load_t5_data
        import torch
        
        # Get a small batch
        train_loader, _, _ = load_t5_data(2, 2, use_schema=True, use_preprocessed=True)
        batch = next(iter(train_loader))
        
        # Simulate what train_epoch does
        encoder_input, encoder_mask, decoder_targets, _ = batch
        
        print(f"✓ Unpacking works as expected")
        print(f"  encoder_input: {encoder_input.shape}")
        print(f"  encoder_mask: {encoder_mask.shape}")  
        print(f"  decoder_targets: {decoder_targets.shape}")
        
        # Check shapes are reasonable
        batch_size = encoder_input.shape[0]
        if encoder_mask.shape[0] != batch_size:
            print(f"✗ ERROR: encoder_mask batch size mismatch")
            return False
        
        if decoder_targets.shape[0] != batch_size:
            print(f"✗ ERROR: decoder_targets batch size mismatch")
            return False
        
        print(f"✓ Batch sizes are consistent")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_preprocessed_data_quality():
    """Check if preprocessed data looks reasonable"""
    print("\n" + "="*80)
    print("CHECKING PREPROCESSED DATA QUALITY")
    print("="*80)
    
    try:
        with open("data_preprocessed/train.nl") as f:
            nl_lines = [l.strip() for l in f.readlines() if l.strip()]
        
        with open("data_preprocessed/train.sql") as f:
            sql_lines = [l.strip() for l in f.readlines() if l.strip()]
        
        print(f"✓ Preprocessed data loaded")
        print(f"  NL lines: {len(nl_lines)}")
        print(f"  SQL lines: {len(sql_lines)}")
        
        if len(nl_lines) != len(sql_lines):
            print(f"✗ ERROR: Mismatched line counts!")
            return False
        
        # Check first few examples
        print(f"\nFirst 3 examples:")
        for i in range(min(3, len(nl_lines))):
            print(f"\n[{i+1}] NL:  {nl_lines[i][:80]}...")
            print(f"    SQL: {sql_lines[i][:80]}...")
        
        # Check if all NL start with "list" (from preprocessing)
        non_list = [nl for nl in nl_lines[:100] if not nl.lower().startswith("list")]
        if non_list:
            print(f"\n⚠️  Warning: {len(non_list)} examples don't start with 'list'")
            print(f"    Example: {non_list[0]}")
        else:
            print(f"\n✓ All samples start with 'list' (preprocessing worked)")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def check_tokenization():
    """Check tokenization is working correctly"""
    print("\n" + "="*80)
    print("CHECKING TOKENIZATION")
    print("="*80)
    
    try:
        from transformers import T5TokenizerFast
        
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        
        # Test NL tokenization
        nl_sample = "translate to SQL: list flights from boston to denver"
        nl_tokens = tokenizer(nl_sample, return_tensors="pt")
        print(f"✓ NL tokenization works")
        print(f"  Sample: {nl_sample}")
        print(f"  Tokens: {nl_tokens['input_ids'].shape[1]}")
        
        # Test SQL tokenization
        sql_sample = "SELECT * FROM flight WHERE city = 'BOSTON'"
        sql_tokens = tokenizer(sql_sample, return_tensors="pt")
        print(f"\n✓ SQL tokenization works")
        print(f"  Sample: {sql_sample}")
        print(f"  Tokens: {sql_tokens['input_ids'].shape[1]}")
        
        # Check vocab
        print(f"\n✓ Tokenizer vocab size: {len(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("T5 TEXT-TO-SQL PRE-TRAINING VALIDATION")
    print("="*80)
    
    checks = [
        ("Data Files", check_data_files),
        ("Preprocessed Data Quality", check_preprocessed_data_quality),
        ("Tokenization", check_tokenization),
        ("Model Loading", check_model_loading),
        ("Data Loader", validate_data_loader),
        ("Training Loop Format", test_training_loop_format),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL CHECKS PASSED - SAFE TO TRAIN ✓✓✓")
        print("="*80)
        print("\nRecommended command:")
        print("python train_t5.py \\")
        print("  --finetune \\")
        print("  --experiment_name 'fixed_v1' \\")
        print("  --learning_rate 1e-4 \\")
        print("  --batch_size 16 \\")
        print("  --max_n_epochs 30 \\")
        print("  --num_beams 4 \\")
        print("  --eval_every 2 \\")  # Changed from 5!
        print("  --patience_epochs 3 \\")
        print("  --use_preprocessed \\")
        print("  --use_schema")
        print("\nNote: Changed eval_every to 2 for faster feedback!")
        return 0
    else:
        print("✗✗✗ SOME CHECKS FAILED - FIX BEFORE TRAINING ✗✗✗")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())