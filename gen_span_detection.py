import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

set_seed(42)


#  MedDec Category Descriptions

CATEGORY_DESCRIPTIONS = [
    "Contact related: Decision regarding admittance or discharge from hospital, "
    "scheduling of control, and referral to other parts of the healthcare system.",
    "Gathering additional information: Decision to obtain information from sources "
    "other than patient interview, physical examination, and patient chart.",
    "Defining problem: Complex, interpretative assessments that define what the "
    "problem is and reflect a medically informed conclusion.",
    "Treatment goal: Decision to set a defined goal for treatment, being more "
    "specific than giving general advice.",
    "Drug: Decision to start, refrain from, stop, alter, or maintain a drug regimen.",
    "Therapeutic procedure: Decision to intervene on a medical problem — plan, "
    "perform, or refrain from therapeutic procedures of a medical nature.",
    "Evaluating test result: Simple, normative assessments of clinical findings "
    "and test results.",
    "Deferment: Decision to actively delay a decision or a rejection to decide on "
    "a problem presented by the patient.",
    "Advice and precaution: Decision to give the patient advice or precaution, "
    "thereby transferring responsibility for action from provider to patient.",
]

NUM_CATEGORIES = len(CATEGORY_DESCRIPTIONS)   # 9
FLAN_T5_MAX_SOURCE = 1024 # As FLAN-T5 was trained on 512 tokens, we extend to 1024 to compromise (rest of the text to be truncated)


# Helper functions

def group_annotations(annotations: list) -> dict:
    """Group annotation list by category"""

    grouped = defaultdict(list)
    for ann in annotations:
        cat_str = ann.get("category", "")
        cat = _parse_cat_number(cat_str)
        if cat is None or cat > 9:
            continue
        grouped[cat].append(ann["decision"].strip())
    return dict(grouped)


def _parse_cat_number(cat_string: str):
    """Extract integer from 'Category 5: Drug' → 5."""

    for i, ch in enumerate(cat_string):
        if ch.isdigit():
            if i + 1 < len(cat_string) and cat_string[i + 1].isdigit():
                return int(cat_string[i : i + 2])
            return int(ch)
    return None


def get_demo(annotations: dict, target_cat: int):
    """
    Pick the in-context example from the same note for one-shot prompting.
    Chooses the category (other than the target category target_cat) with the most annotated spans
    """

    candidates = {k: v for k, v in annotations.items() if k != target_cat and v}
    if not candidates:
        return None, None
    demo_cat = max(candidates, key=lambda k: len(candidates[k]))
    demo_text = "\n".join(f'* "{d}"' for d in candidates[demo_cat])
    return demo_cat, demo_text


# PROMPTING FUNCTIONS

def _prompt_seq2seq(note: str, cat_idx_0: int, demo_text: str = None, demo_cat_idx_0: int = None) -> str:
    """Plain-text/Direct Instruction prompt for encoder-decoder models (FLAN-T5)"""

    cat_desc = CATEGORY_DESCRIPTIONS[cat_idx_0]

    if demo_text:
        # One-shot: show an example first, labelled with the correct demo category
        demo_cat_desc = CATEGORY_DESCRIPTIONS[demo_cat_idx_0] if demo_cat_idx_0 is not None else "a medical decision category"
        prompt = (
            f"Task: Extract all substrings from a clinical note that represent "
            f"medical decisions of the specified category. "
            f"Print each decision on a new line. If none exist, output 'None'.\n\n"
            f"Example category: {demo_cat_desc}\n"
            f"Example decisions:\n{demo_text}\n\n"
            f"Now extract from the following note.\n"
            f"Category: {cat_desc}\n"
            f"Clinical note: {note}\n"
            f"Decisions:"
        )
    else:
        # Zero shot
        prompt = (
            f"Extract all substrings from the following clinical note that represent "
            f"medical decisions of the category '{cat_desc}'. "
            f"Print each decision on a new line. If none exist, output 'None'.\n\n"
            f"Clinical note: {note}\n\n"
            f"Decisions:"
        )
    return prompt


def _prompt_causal(note: str, cat_idx_0: int, tokenizer, demo_cat_idx_0: int = None, demo_text: str = None):
    """Chat-template prompt for causal LLMs (Llama-3)"""
    
    cat_desc = CATEGORY_DESCRIPTIONS[cat_idx_0]

    system_msg = (
        "Extract all substrings from the following clinical note that contain "
        "medical decisions within the specified category.\n"
        "Print each substring on a new line.\n"
        "If no such substring exists, output \"None\".\n"
        f"[Clinical Note]: {note}"
    )

    if demo_text and demo_cat_idx_0 is not None:
        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": f"[Category]: {CATEGORY_DESCRIPTIONS[demo_cat_idx_0]}"},
            {"role": "assistant", "content": demo_text},
            {"role": "user",      "content": f"[Category]: {cat_desc}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"[Category]: {cat_desc}"},
        ]

    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )


# MODEL LOADING

def load_model(model_name: str):
    """Load the LLM and tokenizer (Auto-detects seq2seq vs. causal)"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device} ...")

    config = AutoConfig.from_pretrained(model_name)
    is_seq2seq = getattr(config, "is_encoder_decoder", False)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if is_seq2seq:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model_type = "seq2seq"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model_type = "causal"

    model.eval()
    print(f"  Model type : {model_type}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer, model_type


# GENERATION

def _generate_seq2seq(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", max_length=FLAN_T5_MAX_SOURCE, truncation=True,).to(device)
    
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def _generate_causal(model, tokenizer, input_ids, max_new_tokens: int = 256) -> str:
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        terminators.append(eot_id)

    with torch.no_grad():
        out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9,)
    response_ids = out_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# OUTPUT PREPARATION

def parse_output(raw_text: str) -> list:
    """
    Turn raw LLM output into a clean list of decision strings. 
    - Strip bullet markers, surrounding quotes, and numbering.
    - Drop lines that are empty or say "None".
    """
    
    lines = raw_text.split("\n")
    cleaned = []
    
    for line in lines:
        line = line.strip()
        # Remove common list markers: "1. ", "* ", "- ", "• "
        for marker in ("* ", "- ", "• "):
            if line.startswith(marker):
                line = line[len(marker):]
        # Remove leading numbering like "1. " or "2) "
        if len(line) >= 3 and line[0].isdigit() and line[1] in ".)" and line[2] == " ":
            line = line[3:]
        # Remove surrounding quotes
        line = line.strip('"\'').strip()
        if line and line.lower() not in ("none", "none.", "n/a"):
            cleaned.append(line)
    return cleaned


# MAIN PIPELINE

def run_pipeline(meddec_dir, splits_dir, model_name = "google/flan-t5-small", output_dir = "gens/test_zero_shot",
    split = "test", mode = "zero_shot", max_samples  = None, max_new_tokens = 256,):

    """
    Run pipeline for zero-shot or one-shot LLM-based decision span extraction from medical notes
    Saves the detected spans along with their categories as JSON outputs into output_dir
    """
    meddec_dir = Path(meddec_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # output_dir is on Drive — persists across Colab sessions

    # Read the list of note filenames
    filenames = [ln.strip() for ln in (splits_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_samples:
        filenames = filenames[:max_samples]  # useful for quick smoke tests

    # Load the model
    model, tokenizer, model_type = load_model(model_name)
    device = next(model.parameters()).device if model_type == "seq2seq" else None

    skipped = 0 # counter for skipped notes

    # Loop through the files
    for fname in tqdm(filenames, desc=f"{mode} | {split}"):
        stem     = Path(fname).stem
        out_path = output_dir / f"{stem}.json"

        # If we already have output for this note, skip and move on to the next note
        if out_path.exists():
            continue

        json_path = meddec_dir / "data"     / fname
        txt_path  = meddec_dir / "raw_text" / f"{stem}.txt"

        if not json_path.exists() or not txt_path.exists():
            skipped += 1
            continue

        note_text   = txt_path.read_text(encoding="utf-8")
        ann_data    = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = group_annotations(ann_data.get("annotations", []))  # used only for one-shot demo selection

        predictions = {}

        try:
            # Query the model once per category (i.e. 9 calls per note)
            for cat_1 in range(1, NUM_CATEGORIES + 1):
                cat_0 = cat_1 - 1  # 0-indexed for CATEGORY_DESCRIPTIONS list

                # For one-shot prompting, pick an example annotation from a different category in the same note
                if mode == "one_shot":
                    demo_cat_1, demo_text = get_demo(annotations, target_cat=cat_1)
                    demo_cat_0 = (demo_cat_1 - 1) if demo_cat_1 is not None else None
                else:
                    demo_text  = None
                    demo_cat_0 = None

                # Build prompt and generate the output
                if model_type == "seq2seq":
                    prompt_text = _prompt_seq2seq(note_text, cat_0, demo_text=demo_text, demo_cat_idx_0=demo_cat_0)
                    raw = _generate_seq2seq(model, tokenizer, prompt_text, max_new_tokens)
                else:
                    input_ids = _prompt_causal(note_text, cat_0, tokenizer, demo_cat_idx_0=demo_cat_0, demo_text=demo_text,).to(device or next(model.parameters()).device)
                    raw = _generate_causal(model, tokenizer, input_ids, max_new_tokens)

                # Parse raw LLM output into a clean list of decision strings
                predictions[str(cat_1)] = parse_output(raw)

        # Error handling if note is too long for available RAM
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM on {fname} — skipping.")
            skipped += 1
            continue

        # Generate the output for this note: write all 9 categories + their decision spans to a JSON file
        out_path.write_text(json.dumps({"file_name": fname, "predictions": predictions}, indent=2),encoding="utf-8",)

    print(f"\nDone. Skipped {skipped} files. Results in {output_dir}")
