from __future__ import annotations

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from services.tokens.tokenizer import Vocabulary
from services.models.transformer import MidiTransformer, MidiTransformerConfig
from services.data.dataset import MidiTokenDataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch.get("attn_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="val"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch.get("attn_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=str, default=str(Path(__file__).resolve().parents[1] / "dataset_index.csv"))
    parser.add_argument("--vocab_path", type=str, default=str(Path(__file__).resolve().parents[1] / "features" / "vocab.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary()
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

    # Warm-up pass to build vocabulary on a small subset
    warm_ds = MidiTokenDataset(args.index_csv, vocab=vocab, split="train", max_len=args.max_seq_len)
    warm_loader = DataLoader(warm_ds, batch_size=64, shuffle=True, num_workers=0)
    for i, batch in enumerate(warm_loader):
        # Trigger tokenization to populate vocab, then stop early
        if i > 10:
            break
    # Freeze vocab and persist
    vocab.freeze()
    vocab.save(vocab_path)

    # Build final datasets with frozen vocab
    train_ds = MidiTokenDataset(args.index_csv, vocab=vocab, split="train", max_len=args.max_seq_len)
    val_ds = MidiTokenDataset(args.index_csv, vocab=vocab, split="val", max_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    config = MidiTransformerConfig(
        vocab_size=len(vocab.id_to_token),
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )
    model = MidiTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")
        # Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "config": config.__dict__,
                "vocab": train_ds.tokenizer.vocab.id_to_token,
            }
            torch.save(ckpt, save_dir / "midi_transformer.pt")


if __name__ == "__main__":
    main()


