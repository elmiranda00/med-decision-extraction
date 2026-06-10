import argparse
from pathlib import Path

import torch

from dataset import MedDecDataset, NUM_CATEGORIES, NUM_LABELS, LABEL_O, LABEL_PAD, load_electra_tokenizer
from model import MedDecModel


MAX_LEN = 512   # ELECTRA's positional-embedding limit

CATEGORY_NAMES = ["Contact-related",
    "Gathering information",
    "Defining problem",
    "Treatment goal",
    "Drug",
    "Therapeutic procedure",
    "Evaluating test result",
    "Deferment",
    "Advice and precaution",]


# BIO DECODING

def bio_decode(labels: list) -> list:
    """
    Convert a flat sequence of BIO label indices into a list of (category, start, end) spans.
    - B-n label starts a new span of category n.
    - I-n  label continues the current span IF it matches category n
    - O / PAD :closes any open span

    Robustness: an I-n that appears without a matching B-n is treated as a new B-n (to avoid silently dropping valid spans due to a single mispredicted B token).

    Args:
        labels: list of integer label indices

    Returns: list of (category: int, start: int, end: int)
            category  — 0-indexed (0..8)
            start     — inclusive token index
            end       — exclusive token index  (span = tokens[start:end])
    """
    spans       = []
    current_cat = None
    current_start = None

    for i, label in enumerate(labels):
        
        # Close any open span
        if label == LABEL_PAD or label == LABEL_O:    
            if current_cat is not None:
                spans.append((current_cat, current_start, i))
                current_cat = None
        
        # B-tag: always starts a new span (close previous if open)
        elif label % 2 == 0:
            if current_cat is not None:
                spans.append((current_cat, current_start, i))
            current_cat   = label // 2
            current_start = i
        
        # I-tag
        else:
            
            cat = label // 2
            if current_cat == cat:
                pass   # extend current span — do nothing
            else:
                # Category mismatch or no open span: close old, start new
                if current_cat is not None:
                    spans.append((current_cat, current_start, i))
                current_cat   = cat
                current_start = i

    # Close any span still open at end of sequence
    if current_cat is not None:
        spans.append((current_cat, current_start, len(labels)))

    return spans


# CHUNKED INFERENCE

def predict_full_note(model, input_ids: list, attention_mask: list, device, max_len: int = MAX_LEN) -> list:
    """
    Split a note into non-overlapping chunks (of length MAX_LEN = 512 tokens, ELECTRA limit)
    Concatenate the argmax predictions.

    Args:
        model:         MedDecModel in eval mode
        input_ids:     full token ID list (no length limit)
        attention_mask: corresponding mask list
        device:        torch device
        max

    Returns:
        preds: list of predicted label indices, same length as input_ids
    """
    model.eval()
    all_preds = []

    for start in range(0, len(input_ids), max_len):
        chunk_ids  = torch.tensor(input_ids [start : start + max_len], dtype=torch.long).unsqueeze(0).to(device)
        chunk_mask = torch.tensor(attention_mask[start : start + max_len], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(chunk_ids, chunk_mask)   # [1, chunk_len, 19]

        chunk_preds = logits.argmax(-1)[0].cpu().tolist()
        all_preds.extend(chunk_preds)

    return all_preds


# COMPUTATION OF SPAN-LEVEL F1

def compute_span_f1(gold_spans: set, pred_spans: set) -> dict:
    """
    Compute precision, recall, and F1 from two sets of (sample_idx, cat, start, end) tuples.
    True positive only if ALL four fields match exactly.
    """
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans) - tp
    fn = len(gold_spans) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(model, dataset: MedDecDataset, device, max_len: int = MAX_LEN) -> dict:
    """
    Run full span-level evaluation over a MedDecDataset split.
    Returns a dict with overall and per category metrics
    """
    model.eval()

    all_gold: set = set()   # (sample_idx, cat, start, end)
    all_pred: set = set()

    for sample_idx, sample in enumerate(dataset.samples):

        # Gold spans (human annotated) from the pre-built label sequence
        gold_labels = sample["labels"]
        for cat, start, end in bio_decode(gold_labels):
            all_gold.add((sample_idx, cat, start, end))

        # Predicted spans from chunked model inference
        preds = predict_full_note(model, sample["input_ids"], sample["attention_mask"], device, max_len)
        for cat, start, end in bio_decode(preds):
            all_pred.add((sample_idx, cat, start, end))

    # Calc overall metrics
    overall = compute_span_f1(all_gold, all_pred)
    overall["n_gold"] = len(all_gold)
    overall["n_pred"] = len(all_pred)

    # Calc per-category metrics
    per_cat = {}
    for cat in range(NUM_CATEGORIES):
        cat_gold = {s for s in all_gold if s[1] == cat}
        cat_pred = {s for s in all_pred if s[1] == cat}
        metrics  = compute_span_f1(cat_gold, cat_pred)
        metrics["n_gold"] = len(cat_gold)
        metrics["n_pred"] = len(cat_pred)
        per_cat[cat] = metrics

    return {"overall": overall, "per_cat": per_cat}


# PRINT RESULTS FUNCTION

def print_results(results: dict, split_name: str = "test") -> None:
    """Print a formatted evaluation report."""
    
    ov = results["overall"]
    print(f"\n{'='*60}")
    print(f"  Span-level F1 — {split_name.upper()} SET")
    print(f"{'='*60}")
    print(f"  Gold spans : {ov['n_gold']:5d}")
    print(f"  Pred spans : {ov['n_pred']:5d}")
    print(f"  TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")
    print(f"  Precision  : {ov['precision']:.4f}")
    print(f"  Recall     : {ov['recall']:.4f}")
    print(f"  F1         : {ov['f1']:.4f}  ← main metric")
    print(f"\n  {'Cat':>3}  {'Name':25s}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Gold':>5}  {'Pred':>5}")
    print(f"  {'-'*60}")
    for cat, m in results["per_cat"].items():
        print(
            f"  {cat+1:>3}  {CATEGORY_NAMES[cat]:25s}  "
            f"{m['f1']:>6.3f}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
            f"{m['n_gold']:>5d}  {m['n_pred']:>5d}"
        )
    print(f"{'='*60}\n")


# ENTRY POINT

def run_evaluation(checkpoint_path, meddec_dir, splits_dir, model_name = "google/electra-base-discriminator", 
                   split = "test", max_len = MAX_LEN,):
    """Load a trained model checkpoint and evaluate on the test set"""
    
    checkpoint_path = Path(checkpoint_path)
    meddec_dir      = Path(meddec_dir)
    splits_dir      = Path(splits_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenization and dataset prep
    tokenizer = load_electra_tokenizer(model_name)
    dataset   = MedDecDataset(splits_dir / f"{split}.txt", meddec_dir, tokenizer, train=False, max_len=max_len,)
    print(f"Loaded {len(dataset)} notes from {split} split.")

    # Reconstruct model architecture and load saved weights
    model = MedDecModel(model_name, num_labels=NUM_LABELS).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Evals and print results
    results = evaluate(model, dataset, device, max_len)
    print_results(results, split_name=split)
    return results


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate ELECTRA span detection")
    p.add_argument("--checkpoint",  required=True,  help="Path to best_model.pt")
    p.add_argument("--meddec_dir",  required=True,  help="Path to meddec-mimic-iii/")
    p.add_argument("--splits_dir",  required=True,  help="Path to splits/ folder")
    p.add_argument("--model_name",  default="google/electra-base-discriminator")
    p.add_argument("--split",       default="test",  choices=["train", "val", "test"])
    p.add_argument("--max_len",     type=int, default=MAX_LEN)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(checkpoint_path = args.checkpoint, meddec_dir = args.meddec_dir, splits_dir  = args.splits_dir, 
                   model_name = args.model_name, split = args.split, max_len = args.max_len,)
