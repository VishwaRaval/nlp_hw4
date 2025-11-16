import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

random.seed(0)

# Simple QWERTY neighbor map for a subset of common letters
KEYBOARD_NEIGHBORS = {
    "a": ["q", "w", "s", "z"],
    "s": ["a", "w", "e", "d", "x"],
    "d": ["s", "e", "r", "f", "c"],
    "f": ["d", "r", "t", "g", "v"],
    "e": ["w", "s", "d", "r"],
    "r": ["e", "d", "f", "t"],
    "t": ["r", "f", "g", "y"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
    "o": ["i", "p"],
    "i": ["u", "o"],
}


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    text = example["text"]

    # Tokenize into words
    tokens = word_tokenize(text)
    new_tokens = []

    for tok in tokens:
        # Only transform alphabetic tokens (skip punctuation, numbers, etc.)
        if tok.isalpha():
            lower_tok = tok.lower()
            r = random.random()
            new_tok = tok

            # --- 1) Synonym replacement with probability 0.25 ---
            if r < 0.25:
                synsets = wordnet.synsets(lower_tok)
                synonyms = set()

                for syn in synsets:
                    for lemma in syn.lemmas():
                        name = lemma.name()
                        # Keep only single-word, alphabetic, different synonyms
                        if (
                            name.lower() != lower_tok
                            and "_" not in name
                            and " " not in name
                            and name.isalpha()
                        ):
                            synonyms.add(name)

                if synonyms:
                    repl = random.choice(list(synonyms))
                    # Preserve capitalization of the original token
                    if tok[0].isupper():
                        repl = repl.capitalize()
                    new_tok = repl

            # --- 2) Typo injection with probability 0.15 (0.25 <= r < 0.40) ---
            elif r < 0.40:
                chars = list(tok)
                # indices where we have a neighbor definition
                candidate_indices = [
                    i for i, ch in enumerate(chars) if ch.lower() in KEYBOARD_NEIGHBORS
                ]

                if candidate_indices:
                    idx = random.choice(candidate_indices)
                    neighbors = KEYBOARD_NEIGHBORS[chars[idx].lower()]
                    if neighbors:
                        new_char = random.choice(neighbors)
                        # Preserve original case
                        if chars[idx].isupper():
                            new_char = new_char.upper()
                        chars[idx] = new_char
                        new_tok = "".join(chars)

            new_tokens.append(new_tok)
        else:
            # Non-alphabetic tokens remain unchanged
            new_tokens.append(tok)

    # Detokenize back to a string
    detok_text = TreebankWordDetokenizer().detokenize(new_tokens)
    example["text"] = detok_text

    ##### YOUR CODE ENDS HERE ######

    return example

