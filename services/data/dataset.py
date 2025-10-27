from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
import pretty_midi as pm

from services.tokens.tokenizer import MidiTokenizer, Vocabulary


class MidiTokenDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        tokens_dir: str | Path | None = None,
        vocab_path: str | Path | None = None,
        vocab: Vocabulary | None = None,
        split: str = "train",
        max_len: int = 1024,
    ):
        import pandas as pd

        self.index_csv = Path(index_csv)
        self.tokens_dir = Path(tokens_dir) if tokens_dir else None
        self.max_len = max_len
        self.split = split
        self.df = pd.read_csv(self.index_csv)
        self.df = self.df[self.df["bar_count"] == 2]
        self.df = self.df[self.df["beat_count"] == 4]

        if vocab is not None:
            self.vocab = vocab
        elif vocab_path and Path(vocab_path).exists():
            self.vocab = Vocabulary.load(vocab_path)
        else:
            self.vocab = Vocabulary()
        self.tokenizer = MidiTokenizer(self.vocab)

        # train/val/test split by loop_index hash
        def split_bucket(loop_index: str) -> str:
            h = hash(str(loop_index)) % 100
            if h < 80:
                return "train"
            elif h < 90:
                return "val"
            return "test"

        self.df["split"] = self.df["loop_index"].astype(str).map(split_bucket)
        self.df = self.df[self.df["split"] == self.split]

    def __len__(self) -> int:
        return len(self.df)

    def _load_midi(self, row: Dict[str, Any]) -> pm.PrettyMIDI:
        # In absence of MIDI files, synthesize a simple placeholder melody following bpm
        bpm = int(row.get("bpm", 120) or 120)
        midi = pm.PrettyMIDI(initial_tempo=bpm)
        inst = pm.Instrument(program=0, is_drum=False)
        step_sec = 60.0 / bpm / 16.0
        t = 0.0
        for i in range(0, 8):  # 8 notes over 2 bars
            start = t
            end = t + step_sec * 4
            pitch = 60 + (i % 7)
            inst.notes.append(pm.Note(velocity=96, pitch=pitch, start=start, end=end))
            t += step_sec * 8
        midi.instruments.append(inst)
        return midi

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx].to_dict()
        conditions = {
            "scale": row.get("scale", "C major"),
            "bpm": row.get("bpm", 120),
            "instrument_type": row.get("instrument_type", ""),
            "genre": row.get("genre", ""),
        }
        midi = self._load_midi(row)
        token_ids = self.tokenizer.encode(midi, conditions=conditions)
        # Truncate/pad to max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len]
        attn_mask = torch.zeros(len(token_ids), dtype=torch.bool)
        return {
            "input_ids": torch.tensor(token_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(token_ids[1:], dtype=torch.long),
            "attn_mask": attn_mask[:-1],
        }


