import os
import re

INPUT_DIR = "data"
OUTPUT_DIR = "data_preprocessed"


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_nl_query(q: str) -> str:
    """
    Text-only preprocessing for natural language queries.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Collapse multiple internal spaces to a single space.
    3. Remove trailing punctuation such as '?', '.', ',', ';', ':', '!'.
    4. Add a T5-style task prefix: 'translate English to SQL: '.
    """
    q = q.strip()
    q = re.sub(r"\s+", " ", q)            # collapse whitespace
    q = re.sub(r"[?.,;:!]+$", "", q)      # remove trailing punctuation
    q = q.strip()
    # T5 task prefix
    prefixed = f"translate English to SQL: {q}"
    return prefixed


def normalize_sql_query(s: str) -> str:
    """
    Minimal SQL preprocessing:
    1. Strip leading/trailing whitespace.
    2. Collapse multiple spaces to a single space.
    3. Remove trailing semicolon if present.
    """
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def preprocess_split(split_name: str, has_sql: bool = True):
    """
    Preprocess one split (train/dev/test).

    For train/dev:
      - Read <split>.nl and <split>.sql (if has_sql=True).
    For test:
      - Only <split>.nl.

    Saves preprocessed files under OUTPUT_DIR with the same filenames.
    """
    in_nl_path = os.path.join(INPUT_DIR, f"{split_name}.nl")
    out_nl_path = os.path.join(OUTPUT_DIR, f"{split_name}.nl")

    nl_raw = read_lines(in_nl_path)
    nl_proc = [normalize_nl_query(q) for q in nl_raw]
    write_lines(out_nl_path, nl_proc)
    print(f"[{split_name}] NL: {len(nl_proc)} examples -> {out_nl_path}")

    if has_sql:
        in_sql_path = os.path.join(INPUT_DIR, f"{split_name}.sql")
        out_sql_path = os.path.join(OUTPUT_DIR, f"{split_name}.sql")
        sql_raw = read_lines(in_sql_path)
        assert len(sql_raw) == len(nl_raw), "NL and SQL files must have same length"
        sql_proc = [normalize_sql_query(s) for s in sql_raw]
        write_lines(out_sql_path, sql_proc)
        print(f"[{split_name}] SQL: {len(sql_proc)} examples -> {out_sql_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("T5 DATA PREPROCESSING (no augmentation)")
    print("=" * 80)

    # Train & dev have SQL
    preprocess_split("train", has_sql=True)
    preprocess_split("dev", has_sql=True)

    # Test has only NL queries
    preprocess_split("test", has_sql=False)

    print("\nAll preprocessed files written to:", OUTPUT_DIR)
    print("You can now run compute_t5_statistics.py to fill Tables 1 & 2.")


if __name__ == "__main__":
    main()
