import os
import re

INPUT_DIR = "data"
OUTPUT_DIR = "data_preprocessed"


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def canonicalize_question(text: str) -> str:
    """
    Preprocess a natural language query so that queries have a uniform form.

    Steps:
      1. Lowercase
      2. Normalize leading phrases (give me, show me, what..., can you..., etc.) to start with "list "
      3. Collapse extra whitespace
      4. Remove punctuation at the end
      5. Ensure the query begins with "list "
    """
    # Step 1: lowercase first to simplify regex handling
    q = text.lower()

    # Handle leading polite / request phrases in grouped patterns

    # Direct request patterns: "give me", "show me", "provide me", ...
    direct_openings = [
        r"give me",
        r"show me",
        r"provide me",
        r"tell me",
        r"find me",
        r"get me",
    ]
    direct_pattern = r"^(" + "|".join(direct_openings) + r")(\s+the)?\s+"
    q = re.sub(direct_pattern, "list ", q)

    # Desire patterns: "i want", "i need", "i would like", "i'd like"
    desire_openings = [
        r"i want",
        r"i need",
        r"i would like",
        r"i'd like",
    ]
    desire_pattern = r"^(" + "|".join(desire_openings) + r")(\s+the)?\s+"
    q = re.sub(desire_pattern, "list ", q)

    # "what" / "which" questions, e.g. "what is the", "which are the"
    q = re.sub(
        r"^(what|which)(\s+is|\s+are)?(\s+the)?\s+",
        "list ",
        q,
    )

    # "can you" / "could you" questions
    q = re.sub(
        r"^(can you|could you)\s+",
        "list ",
        q,
    )

    # Step 3: collapse multiple spaces
    q = re.sub(r"\s+", " ", q).strip()

    # Step 4: strip trailing punctuation like ?,.,;:!
    q = re.sub(r"[?.,;:!]+$", "", q).strip()

    # Step 5: make sure everything still starts with "list "
    # If no earlier pattern matched, prepend "list "
    if not q.startswith("list "):
        q = "list " + q.lstrip()

    return q


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

    Writes preprocessed files under OUTPUT_DIR with the same filenames.
    """
    # Natural language side
    in_nl_path = os.path.join(INPUT_DIR, f"{split_name}.nl")
    out_nl_path = os.path.join(OUTPUT_DIR, f"{split_name}.nl")

    nl_raw = read_lines(in_nl_path)
    nl_proc = [canonicalize_question(q) for q in nl_raw]
    write_lines(out_nl_path, nl_proc)
    print(f"[{split_name}] NL: {len(nl_proc)} examples -> {out_nl_path}")

    # SQL side (if present)
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
    print("T5 DATA PREPROCESSING")
    print("=" * 80)

    # Train & dev have SQL
    preprocess_split("train", has_sql=True)
    preprocess_split("dev", has_sql=True)

    # Test has only NL queries
    preprocess_split("test", has_sql=False)

    print("\n" + "=" * 80)
    print("✓ All preprocessed files written to:", OUTPUT_DIR)
    print("✓ All NL queries now begin with: 'list ...'")
    print("=" * 80)


if __name__ == "__main__":
    main()