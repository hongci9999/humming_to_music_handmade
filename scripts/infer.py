from __future__ import annotations

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path

import torch
import pretty_midi as pm

from services.tokens.tokenizer import Vocabulary, MidiTokenizer
from services.models.transformer import MidiTransformer, MidiTransformerConfig


def build_condition_tokens(tokenizer: MidiTokenizer, conditions):
    # Use an empty MIDI to just produce the header prefix and program
    midi = pm.PrettyMIDI(initial_tempo=int(conditions.get("bpm", 120)))
    inst = pm.Instrument(program=0, is_drum=False)
    midi.instruments.append(inst)
    # encode and take only until first BAR token (prefix)
    ids = tokenizer.encode(midi, conditions)
    # cut before BAR_1
    tokens = [tokenizer.vocab.get_token(i) for i in ids]
    cut = 0
    for i, t in enumerate(tokens):
        if t.startswith("BAR_"):
            cut = i
            break
    return ids[:cut]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "midi_transformer.pt"))
    parser.add_argument("--vocab_path", type=str, default=str(Path(__file__).resolve().parents[1] / "features" / "vocab.json"))
    parser.add_argument("--out_midi", type=str, default="sample.mid")
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--scale", type=str, default="C major")
    parser.add_argument("--instrument", type=str, default="Piano")
    parser.add_argument("--genre", type=str, default="Dance")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = MidiTransformerConfig(**ckpt["config"])  # type: ignore[arg-type]
    vocab = Vocabulary()
    vocab.id_to_token = ckpt["vocab"]
    vocab.token_to_id = {t: i for i, t in enumerate(vocab.id_to_token)}
    tokenizer = MidiTokenizer(vocab)

    model = MidiTransformer(config)
    model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
    model.to(device).eval()

    conditions = {"bpm": args.bpm, "scale": args.scale, "instrument": args.instrument, "genre": args.genre}
    cond_ids = build_condition_tokens(tokenizer, conditions)
    input_ids = torch.tensor([cond_ids], dtype=torch.long, device=device)

    # Generate up to a reasonable budget; the tokenizer/decoder enforces 2 bars
    eos_id = tokenizer.vocab.get_id("<EOS>")
    out_ids = model.generate(input_ids, max_new_tokens=512, temperature=1.0, top_p=0.9, eos_token_id=eos_id)
    out_ids_list = out_ids[0].tolist()

    midi = tokenizer.decode(out_ids_list, fallback_bpm=args.bpm)
    Path(args.out_midi).parent.mkdir(parents=True, exist_ok=True)
    midi.write(args.out_midi)
    print(f"Wrote {args.out_midi}")


if __name__ == "__main__":
    main()


