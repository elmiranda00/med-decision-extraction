"""
Phase 2.3 — Training loop for ELECTRA BIO span detection.

Quick-start (Colab):
    python train.py \\
        --meddec_dir  /content/drive/MyDrive/02\\ Data/meddec-mimic-iii \\
        --splits_dir  /content/drive/MyDrive/02\\ Data/meddec-mimic-iii/splits \\
        --output_dir  checkpoints

Tracked in MLflow (run `mlflow ui` locally to browse results).
Checkpoints saved to --output_dir:  best_model.pt  +  last_model.pt
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from dataset import NUM_LABELS, LABEL_PAD, make_dataloader, load_electra_tokenizer
from model import MedDecModel, build_training_objects


# HELPER FUNCTIONS FOR METRICS

def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """Returns the number of corect and total tokens. Ignores PAD positions."""
    
    preds = logits.argmax(-1) # [batch, seq]
    mask = labels != LABEL_PAD  # TRUE for tokens for which loss is computed
    correct = ((preds == labels) & mask).sum().item()
    total   = mask.sum().item()
    
    return correct, total


def validate(model, loader, criterion, device, max_len: int = 512):
    """
    Quick per-epoch validation, gives token-level loss/accuracy.
    Note: Sequences are truncated (max_len = 512) to stay within ELECTRA limit.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"][:, :max_len].to(device)
            mask   = batch["attention_mask"][:, :max_len].to(device)
            labels = batch["labels"][:, :max_len].to(device)

            logits = model(ids, mask) # [B, S, 19]
            loss   = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            total_loss += loss.item()

            c, t  = _token_accuracy(logits, labels)
            correct += c
            total   += t

    model.train()
    avg_loss = total_loss / max(len(loader), 1)
    acc      = correct / total if total > 0 else 0.0
    
    return avg_loss, acc


# MAIN TRAINING FUNCTION

def train(meddec_dir, splits_dir, model_name   = "google/electra-base-discriminator", output_dir   = "checkpoints",
    num_epochs = 5, batch_size = 4, lr = 4e-5, grad_accum  = 2, warmup_ratio = 0.1, max_len = 512, experiment   = "meddec-electra",):
    
    meddec_dir = Path(meddec_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataloaders
    tokenizer    = load_electra_tokenizer(model_name)
    train_loader = make_dataloader(splits_dir / "train.txt", meddec_dir, tokenizer, train=True, batch_size=batch_size, max_len=max_len,)
    val_loader = make_dataloader(splits_dir / "val.txt", meddec_dir, tokenizer, train=False, batch_size=batch_size, max_len=max_len,)
    
    print(f"Train: {len(train_loader.dataset)} notes  |  Val: {len(val_loader.dataset)} notes")

    # Model
    model = MedDecModel(model_name, num_labels=NUM_LABELS).to(device)

    # total optimiser updates = (batches / grad_accum) × epochs
    steps_per_epoch = max(len(train_loader) // grad_accum, 1)
    total_steps     = steps_per_epoch * num_epochs

    criterion, optimizer, scheduler = build_training_objects(model, total_steps, lr=lr, warmup_ratio=warmup_ratio,)

    # MLflow logging
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "model_name":    model_name,
            "num_epochs":    num_epochs,
            "batch_size":    batch_size,
            "lr":            lr,
            "grad_accum":    grad_accum,
            "warmup_ratio":  warmup_ratio,
            "max_len":       max_len,
            "total_steps":   total_steps,
            "train_samples": len(train_loader.dataset),
            "val_samples":   len(val_loader.dataset),
        })

        best_val_loss = float("inf")
        global_step   = 0
        optimizer.zero_grad()

        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            for step, batch in enumerate(pbar):
                ids    = batch["input_ids"].to(device)
                mask   = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                logits = model(ids, mask)  # [B, S, 19]

                # Flatten to [B*S, 19] vs [B*S] for token-level cross-entropy
                loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
                (loss / grad_accum).backward()  # scale before backward

                epoch_losses.append(loss.item())

                # Optimiser step
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log smoothed training loss every 50 optimiser steps
                    if global_step % 50 == 0:
                        window = epoch_losses[-50 * grad_accum:]
                        mlflow.log_metric("train_loss", np.mean(window), step=global_step)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Validation at the end of each epoch
            val_loss, val_acc = validate(model, val_loader, criterion, device, max_len)
            epoch_train_loss  = np.mean(epoch_losses)

            print(f"Epoch {epoch:2d} | train_loss={epoch_train_loss:.4f} | val_loss={val_loss:.4f} | val_token_acc={val_acc:.3f}")
            
            # MLflow logging
            mlflow.log_metrics({
                "epoch_train_loss": epoch_train_loss,
                "val_loss":         val_loss,
                "val_token_acc":    val_acc,}, step=epoch)

            # Save best checkpoint (lowest val loss used)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path     = output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  → Checkpoint saved: {best_path}")
                
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

        # Save final weights too
        torch.save(model.state_dict(), output_dir / "last_model.pt")
        try:
            mlflow.log_artifact(str(output_dir / "best_model.pt"))
        except Exception:
            pass 

    print("Training complete.")
    return model


# CLI entry point

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ELECTRA for MedDec BIO tagging")
    p.add_argument("--meddec_dir",   required=True,  help="Path to meddec-mimic-iii/")
    p.add_argument("--splits_dir",   required=True,  help="Path to folder containing train/val/test .txt files")
    p.add_argument("--model_name",   default="google/electra-base-discriminator")
    p.add_argument("--output_dir",   default="checkpoints")
    p.add_argument("--num_epochs",   type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=4e-5)
    p.add_argument("--grad_accum",   type=int,   default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_len",      type=int,   default=512)
    p.add_argument("--experiment",   default="meddec-electra")
    return p.parse_args()


if __name__ == "__main__":
    
    args = _parse_args()
    train(meddec_dir = args.meddec_dir, splits_dir = args.splits_dir, model_name = args.model_name, output_dir = args.output_dir,
          num_epochs = args.num_epochs, batch_size = args.batch_size, lr = args.lr, grad_accum   = args.grad_accum, 
          warmup_ratio = args.warmup_ratio, max_len = args.max_len, experiment   = args.experiment,)
