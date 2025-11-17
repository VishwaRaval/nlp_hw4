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


def canonicalize_question(q: str) -> str:
    """
    Make the *content* of the question start with `list ...`
    so that all queries have a uniform imperative style.

    Examples:
      "give me the flights from denver"  -> "list the flights from denver"
      "what flights from tacoma..."     -> "list flights from tacoma..."
      "can you show me flights..."      -> "list flights..."
      "pittsburgh to boston saturday"   -> "list pittsburgh to boston saturday"
    """
    # lowercase + basic cleanup
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)             # collapse internal whitespace
    q = re.sub(r"[?.,;:!]+$", "", q)       # remove trailing punctuation
    q = q.strip()

    # Normalise a variety of openings to "list"
    # (order matters: more specific patterns first)
    patterns = [
        r"^(give me|show me|please show me|please give me)\b",
        r"^(can you|could you|would you)\b",
        r"^(i would like to see|i'd like to see)\b",
        r"^(i would like|i'd like|i want|i need)\b",
        r"^(do you have)\b",
        r"^(are there|is there)\b",
        r"^(what is|what are|what)\b",
        r"^(which is|which are|which)\b",
        r"^(how much is|how much are|how much)\b",
        r"^(how many)\b",
    ]

    for pat in patterns:
        # Replace the matched opening with "list"
        q_new = re.sub(pat, "list", q)
        if q_new != q:
            q = q_new
            break

    # If it already starts with "list", just standardize spacing
    if q.startswith("list "):
        q = "list " + q[len("list "):].lstrip()
    elif q == "list":
        # rare case "list" alone
        pass
    else:
        # Fallback: if nothing matched, prepend "list "
        q = "list " + q

    # Final whitespace cleanup
    q = re.sub(r"\s+", " ", q).strip()
    return q


def normalize_nl_query(q: str) -> str:
    """
    Preprocess natural language query

    Steps:
    1. Canonicalize the question to start with "list ..."
    2. Add T5-style task prefix: "translate English to SQL: "
    """
    core = canonicalize_question(q)
    # Now add the T5 task prefix
    prefixed = f"translate English to SQL: {core}"
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
    print("T5 DATA PREPROCESSING (no augmentation, canonical 'list' questions)")
    print("=" * 80)

    # Train & dev have SQL
    preprocess_split("train", has_sql=True)
    preprocess_split("dev", has_sql=True)

    # Test has only NL queries
    preprocess_split("test", has_sql=False)

    print("\nAll preprocessed files written to:", OUTPUT_DIR)
    print("All NL queries now begin with: 'translate English to SQL: list ...'")


if __name__ == "__main__":
    main()
