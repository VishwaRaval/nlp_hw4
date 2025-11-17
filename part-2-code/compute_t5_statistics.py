from transformers import T5TokenizerFast
import numpy as np
import os


MODEL_NAME = "google-t5/t5-small"


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokenize_and_summarize(texts, tokenizer, add_special_tokens=True):
    """
    Tokenize a list of texts with the T5 tokenizer and compute statistics.
    We do NOT truncate here; we want the true length distribution.
    """
    if len(texts) == 0:
        return {
            "num_examples": 0,
            "mean_len": 0.0,
            "median_len": 0.0,
            "std_len": 0.0,
            "max_len": 0,
            "p95_len": 0,
            "vocab_size": 0,
        }

    encodings = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    input_ids_list = encodings["input_ids"]
    lengths = np.array([len(ids) for ids in input_ids_list], dtype=np.int32)

    # Vocabulary used in this split
    vocab_ids = set()
    for ids in input_ids_list:
        vocab_ids.update(ids)

    stats = {
        "num_examples": len(texts),
        "mean_len": float(lengths.mean()),
        "median_len": float(np.median(lengths)),
        "std_len": float(lengths.std()),
        "max_len": int(lengths.max()),
        "p95_len": int(np.percentile(lengths, 95)),
        "vocab_size": len(vocab_ids),
    }
    return stats


def print_split_stats(
    split_name, nl_stats, sql_stats, include_num_examples=True, table_label="TABLE"
):
    print("\n" + "=" * 80)
    print(f"{table_label}: {split_name.upper()} SET")
    print("=" * 80)

    if include_num_examples:
        print(f"Number of examples: {nl_stats['num_examples']}")

    # NL
    print(
        f"Mean sentence length (NL): {nl_stats['mean_len']:.2f} tokens "
        f"(median={nl_stats['median_len']:.2f}, max={nl_stats['max_len']}, p95={nl_stats['p95_len']})"
    )
    # SQL (may be None for test)
    if sql_stats is not None:
        print(
            f"Mean SQL query length: {sql_stats['mean_len']:.2f} tokens "
            f"(median={sql_stats['median_len']:.2f}, max={sql_stats['max_len']}, p95={sql_stats['p95_len']})"
        )
    else:
        print("Mean SQL query length: N/A (no SQL file)")

    print(f"Vocabulary size (natural language): {nl_stats['vocab_size']}")
    if sql_stats is not None:
        print(f"Vocabulary size (SQL): {sql_stats['vocab_size']}")
    else:
        print("Vocabulary size (SQL): N/A (no SQL file)")


def compute_for_split(split_name, base_dir, tokenizer, has_sql=True):
    nl_path = os.path.join(base_dir, f"{split_name}.nl")
    nl_texts = load_lines(nl_path)
    nl_stats = tokenize_and_summarize(nl_texts, tokenizer)

    if has_sql:
        sql_path = os.path.join(base_dir, f"{split_name}.sql")
        sql_texts = load_lines(sql_path)
        assert len(sql_texts) == len(nl_texts), "NL and SQL must have same #examples"
        sql_stats = tokenize_and_summarize(sql_texts, tokenizer)
    else:
        sql_stats = None

    return nl_stats, sql_stats


def main():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    print("=" * 80)
    print("TABLE 1: Data Statistics BEFORE Pre-processing")
    print("=" * 80)
    print(f"\nUsing T5 Tokenizer: {MODEL_NAME}\n")

    # BEFORE preprocessing: from data/
    train_nl_orig, train_sql_orig = compute_for_split(
        "train", "data", tokenizer, has_sql=True
    )
    dev_nl_orig, dev_sql_orig = compute_for_split(
        "dev", "data", tokenizer, has_sql=True
    )

    print_split_stats(
        "train", train_nl_orig, train_sql_orig, include_num_examples=True, table_label="TABLE 1"
    )
    print_split_stats(
        "dev", dev_nl_orig, dev_sql_orig, include_num_examples=True, table_label="TABLE 1"
    )

    print("\n" + "=" * 80)
    print("TABLE 2: Data Statistics AFTER Pre-processing")
    print("=" * 80)
    print(f"\nModel name: {MODEL_NAME}")
    print(
        "Preprocessing: strip + whitespace normalization + trailing punctuation removal "
        "+ task prefix 'translate English to SQL: ' for NL; "
        "whitespace normalization and semicolon removal for SQL.\n"
    )

    if not os.path.isdir("data_preprocessed"):
        print("data_preprocessed/ not found. Run preprocess_t5_data.py first.")
        return

    # AFTER preprocessing: from data_preprocessed/
    train_nl_prep, train_sql_prep = compute_for_split(
        "train", "data_preprocessed", tokenizer, has_sql=True
    )
    dev_nl_prep, dev_sql_prep = compute_for_split(
        "dev", "data_preprocessed", tokenizer, has_sql=True
    )

    # For Table 2 we usually don't repeat "number of examples" (it’s unchanged),
    # but we still know it from the stats if you want to mention it in prose.
    print_split_stats(
        "train", train_nl_prep, train_sql_prep, include_num_examples=False, table_label="TABLE 2"
    )
    print_split_stats(
        "dev", dev_nl_prep, dev_sql_prep, include_num_examples=False, table_label="TABLE 2"
    )

    # (Optional) quick summary of impact
    print("\n" + "=" * 80)
    print("IMPACT OF PREPROCESSING (NL only, mean length & vocab size)")
    print("=" * 80)

    for split, nl_orig, nl_prep in [
        ("TRAIN", train_nl_orig, train_nl_prep),
        ("DEV", dev_nl_orig, dev_nl_prep),
    ]:
        print(f"\n{split}:")
        print(
            f"  Mean NL length: {nl_orig['mean_len']:.2f} -> {nl_prep['mean_len']:.2f} "
            f"(Δ = {nl_prep['mean_len'] - nl_orig['mean_len']:.2f})"
        )
        print(
            f"  NL vocab size: {nl_orig['vocab_size']} -> {nl_prep['vocab_size']} "
            f"(Δ = {nl_prep['vocab_size'] - nl_orig['vocab_size']})"
        )


if __name__ == "__main__":
    main()
