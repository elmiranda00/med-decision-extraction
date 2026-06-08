"""
BIO (Beginning/Inside/Outside) Labelling scheme:
    B-cat_n  =  n * 2          n = 0..8
    I-cat_n  =  n * 2 + 1      n = 0..8
    O        =  18             "outside" — token belongs to no decision span
    padding  =  -100           CrossEntropyLoss ignores this index

Total label classes = 9 * 2 + 1 = 19
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizerFast

NUM_CATEGORIES = 9
NUM_LABELS     = NUM_CATEGORIES * 2 + 1   # 19
LABEL_O        = NUM_CATEGORIES * 2       # 18
LABEL_PAD      = -100  # ignored by CrossEntropyLoss
VALID_CATS     = set(range(NUM_CATEGORIES))

MAX_LEN        = 512  # ELECTRA's positional-embedding limit


# HELPER FUNCTIONS

def parse_category(cat_string):
    """Extract the integer (1-9) corresponding to the category from the MedDec annotations
    e.g. 'Category 5: Drug' -> '5'
    Note: category 10 (legal/insurance) is excluded from modelling and returns None.
    """
    for i, ch in enumerate(cat_string):
        if ch.isdigit():
            # Two-digit check (handles 'Category 10: ...')
            if i + 1 < len(cat_string) and cat_string[i + 1].isdigit():
                cat_1indexed = int(cat_string[i : i + 2])
            else:
                cat_1indexed = int(ch)
            cat_0indexed = cat_1indexed - 1
            return cat_0indexed if cat_0indexed in VALID_CATS else None
    return None


def char_to_token_safe(encoding, char_pos, nudge_forward=True):
    """
    Nudges the character offset forward/backward 
    until it finds and index corresponding to a token (i.e. not a whitespace)

    Note: nudging limit is up to 10 characters away.

    nudge_forward=True, for span START  (look right)
    nudge_forward=False, for span END    (look left)
    """
    direction = 1 if nudge_forward else -1
    for delta in range(10):
        tok = encoding.char_to_token(char_pos + direction * delta)
        if tok is not None:
            return tok
    return None


def build_label_sequence(encoding, annotations, seq_len):
    """
    Turn a list of character-level annotations into a list of labels per token

    Steps:
      1. Every token starts as LABEL_PAD (-100).
      2. Any token that maps to real characters (not CLS/SEP) is set to LABEL_O (18).
      3. For each valid annotation span, the start token gets B (n*2) and
         subsequent tokens get I (n*2+1).

    This means:
      - CLS / SEP → LABEL_PAD  (loss ignores them)
      - tokens with no annotation → LABEL_O
      - annotated tokens → B or I
    """
    
    labels = [LABEL_PAD] * seq_len

    # Mark every real token as O (special tokens keep LABEL_PAD)
    for tok_idx in range(seq_len):
        if encoding.token_to_chars(tok_idx) is not None:
            labels[tok_idx] = LABEL_O

    for annot in annotations:
        cat = parse_category(annot.get("category", ""))
        if cat is None:
            continue

        start_char = int(annot["start_offset"])
        end_char   = int(annot["end_offset"])    # exclusive: span = text[start:end]

        enc_start = char_to_token_safe(encoding, start_char,     nudge_forward=True)
        enc_last  = char_to_token_safe(encoding, end_char - 1,   nudge_forward=False)

        if enc_start is None:
            continue
        enc_last  = enc_last if enc_last is not None else enc_start
        enc_end   = enc_last + 1  # convert to exclusive token index

        if enc_end <= enc_start:
            enc_end = enc_start + 1

        labels[enc_start] = cat * 2  # B
        for t in range(enc_start + 1, min(enc_end, seq_len)):
            labels[t] = cat * 2 + 1  # I

    return labels


# DATASET CLASS

class MedDecDataset(Dataset):
    """
    PyTorch Dataset for MedDec token classification.

    Tokenisation is done once in __init__ 
    __getitem__ applies the windowing strategy for training (since ELECTRA can handle 512 token windows)

    Training  (train=True): A random MAX_LEN-token window is sampled on every call, so the model sees different regions of long notes across epochs.
    Evaluation (train=False): The full token sequence is returned.  evaluate.py is responsible for splitting it into MAX_LEN-token chunks before feeding to the model.
    """

    def __init__(self, split_file, meddec_dir, tokenizer,
                 train=False, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.train     = train
        self.max_len   = max_len
        self.samples   = []

        meddec_dir = Path(meddec_dir)
        filenames  = [
            line.strip()
            for line in Path(split_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        skipped = 0
        for fname in filenames:
            stem     = Path(fname).stem
            json_path = meddec_dir / "data"     / fname
            txt_path  = meddec_dir / "raw_text" / f"{stem}.txt"

            if not json_path.exists() or not txt_path.exists():
                skipped += 1
                continue

            text        = txt_path.read_text(encoding="utf-8")
            annot_data  = json.loads(json_path.read_text(encoding="utf-8"))
            annotations = annot_data.get("annotations", [])

            # Tokenise the FULL note (no truncation; long notes handled by windowing)
            # The fast tokenizer stores internal character alignments used by
            # char_to_token() and token_to_chars().
            encoding = tokenizer(text, add_special_tokens=True, truncation=False)
            seq_len  = len(encoding["input_ids"])

            labels = build_label_sequence(encoding, annotations, seq_len)

            self.samples.append({
                "input_ids":      encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels":         labels,
                "file_name":      fname,
            })

        if skipped:
            print(f"WARNING: skipped {skipped} files (missing .json or .txt)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        ids    = s["input_ids"]
        mask   = s["attention_mask"]
        labels = s["labels"]
        n      = len(ids)

        if self.train and n > self.max_len:
            start  = random.randint(0, n - self.max_len)
            ids    = ids   [start : start + self.max_len]
            mask   = mask  [start : start + self.max_len]
            labels = labels[start : start + self.max_len]

        return {
            "input_ids":      torch.tensor(ids,    dtype=torch.long),
            "attention_mask": torch.tensor(mask,   dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "file_name":      s["file_name"],
        }


# DATALOADER

def make_dataloader(split_file, meddec_dir, tokenizer, train=False, batch_size=4, max_len=MAX_LEN):
    """
    Build a DataLoader from a split file.

    The collate function pads each batch to its longest sequence.
    Padding tokens receive LABEL_PAD so the loss ignores them.
    """
    dataset = MedDecDataset(split_file, meddec_dir, tokenizer, train=train, max_len=max_len)

    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        longest = max(b["input_ids"].shape[0] for b in batch)
        ids_out, mask_out, labels_out, names = [], [], [], []

        for b in batch:
            n   = b["input_ids"].shape[0]
            pad = longest - n
            ids_out.append(
                torch.nn.functional.pad(b["input_ids"],      (0, pad), value=pad_id)
            )
            mask_out.append(
                torch.nn.functional.pad(b["attention_mask"], (0, pad), value=0)
            )
            labels_out.append(
                torch.nn.functional.pad(b["labels"],         (0, pad), value=LABEL_PAD)
            )
            names.append(b["file_name"])

        return {
            "input_ids":      torch.stack(ids_out),
            "attention_mask": torch.stack(mask_out),
            "labels":         torch.stack(labels_out),
            "file_names":     names,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_fn,
    )


# LOAD TOKENIZER

def load_electra_tokenizer(model_name="google/electra-base-discriminator"):

    return ElectraTokenizerFast.from_pretrained(model_name)
