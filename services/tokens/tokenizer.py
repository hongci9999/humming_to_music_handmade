from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pretty_midi as pm


# Grid settings: 4/4, 16 steps per beat → 64 steps per bar → 2 bars = 128 steps
STEPS_PER_BEAT = 16
BEATS_PER_BAR = 4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR  # 64
TOTAL_STEPS = STEPS_PER_BAR * 2  # 128


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def bpm_to_step_seconds(bpm: float) -> float:
    seconds_per_beat = 60.0 / max(bpm, 1e-6)
    return seconds_per_beat / STEPS_PER_BEAT


def parse_scale(scale: str) -> Tuple[str, str]:
    if not scale:
        return ("C", "major")
    s = scale.replace("♭", "b").replace("♯", "#").strip()
    parts = s.split()
    if len(parts) == 2:
        root, mode = parts
        mode = mode.lower()
        mode = "major" if mode.startswith("maj") else ("minor" if mode.startswith("min") else mode)
        return (root.upper(), mode)
    return (s.upper(), "major")


def quantize_time_to_step(time_sec: float, step_sec: float) -> int:
    return max(0, int(round(time_sec / step_sec)))


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: List[str] = field(default_factory=list)
    frozen: bool = False

    def __post_init__(self):
        if not self.token_to_id and not self.id_to_token:
            for t in SPECIAL_TOKENS:
                self._append_token(t)

    @property
    def unk_id(self) -> int:
        return self.token_to_id.get("<UNK>", 0)

    def _append_token(self, token: str) -> int:
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.frozen:
            return self.unk_id
        return self._append_token(token)

    def get_id(self, token: str) -> int:
        return self.add_token(token)

    def get_token(self, idx: int) -> str:
        return self.id_to_token[idx]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id_to_token": self.id_to_token, "frozen": self.frozen}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_token = data["id_to_token"]
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        frozen = bool(data.get("frozen", False))
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, frozen=frozen)

    def freeze(self):
        self.frozen = True


@dataclass
class TokenizerConfig:
    steps_per_beat: int = STEPS_PER_BEAT
    beats_per_bar: int = BEATS_PER_BAR
    include_velocity: bool = True
    velocity_bins: int = 32  # 1..32


class MidiTokenizer:
    def __init__(self, vocab: Optional[Vocabulary] = None, config: Optional[TokenizerConfig] = None):
        self.vocab = vocab or Vocabulary()
        self.config = config or TokenizerConfig()

    # ---------- Token helpers ----------
    def tok_bar(self, bar_idx: int) -> str:
        return f"BAR_{bar_idx}"

    def tok_pos(self, pos: int) -> str:
        return f"POS_{pos}"

    def tok_note_on(self, pitch: int) -> str:
        return f"NOTE_ON_{pitch}"

    def tok_note_off(self, pitch: int) -> str:
        return f"NOTE_OFF_{pitch}"

    def tok_vel(self, vel_bin: int) -> str:
        return f"VEL_{vel_bin}"

    def tok_prog(self, program: int) -> str:
        return f"PROG_{program}"

    def tok_key(self, root: str) -> str:
        return f"KEY_{root}"

    def tok_mode(self, mode: str) -> str:
        return f"MODE_{mode.upper()}"

    def tok_bpm(self, bpm_rounded: int) -> str:
        return f"BPM_{bpm_rounded}"

    def tok_instr(self, instr: str) -> str:
        return f"INSTR_{instr}"

    def tok_genre(self, genre: str) -> str:
        return f"GENRE_{genre}"

    # ---------- Public API ----------
    def encode(self, midi: pm.PrettyMIDI, conditions: Optional[Dict[str, Any]] = None) -> List[int]:
        # Condition prefix
        prefix: List[str] = ["<BOS>"]
        if conditions:
            # key/mode
            scale = conditions.get("scale", "")
            root, mode = parse_scale(scale)
            prefix.append(self.tok_key(root))
            prefix.append(self.tok_mode(mode))
            # bpm
            bpm = int(round(float(conditions.get("bpm", 120))))
            prefix.append(self.tok_bpm(bpm))
            # instrument/genre (optional, sanitized)
            instr = str(conditions.get("instrument_type") or conditions.get("instrument") or "").upper().replace(" ", "_")
            genre = str(conditions.get("genre") or "").upper().replace(" ", "_")
            if instr:
                prefix.append(self.tok_instr(instr))
            if genre:
                prefix.append(self.tok_genre(genre))

        # Choose first non-drum instrument; fall back to program 0
        program = 0
        is_drum = False
        for inst in midi.instruments:
            if not inst.is_drum and len(inst.notes) > 0:
                program = inst.program
                is_drum = False
                break
        else:
            # fallback: if all drums or empty, pick first if present
            if midi.instruments:
                program = midi.instruments[0].program
                is_drum = midi.instruments[0].is_drum

        tokens: List[str] = prefix + [self.tok_prog(program)]

        # Build step grid
        bpm = int(round(float(conditions.get("bpm", 120)))) if conditions else 120
        step_sec = bpm_to_step_seconds(bpm)

        # Gather notes across instruments (merge), ignoring drums unless only drums exist
        merged_notes: List[pm.Note] = []
        non_drum_found = any((not inst.is_drum and inst.notes) for inst in midi.instruments)
        for inst in midi.instruments:
            if non_drum_found and inst.is_drum:
                continue
            merged_notes.extend(inst.notes)

        # Quantize to steps and clip to 2 bars
        on_events: Dict[int, List[int]] = {}
        off_events: Dict[int, List[int]] = {}
        for n in merged_notes:
            on_step = quantize_time_to_step(n.start, step_sec)
            off_step = quantize_time_to_step(n.end, step_sec)
            on_step = max(0, min(on_step, TOTAL_STEPS - 1))
            off_step = max(0, min(off_step, TOTAL_STEPS - 1))
            # Ensure at least 1 step duration
            if off_step <= on_step:
                off_step = min(on_step + 1, TOTAL_STEPS - 1)
            on_events.setdefault(on_step, []).append(n.pitch)
            off_events.setdefault(off_step, []).append(n.pitch)

        # Emit tokens bar by bar, position by position
        for bar in range(1, 3):
            tokens.append(self.tok_bar(bar))
            start_step = (bar - 1) * STEPS_PER_BAR
            end_step = start_step + STEPS_PER_BAR
            for local_pos, step in enumerate(range(start_step, end_step)):
                pos_tok = self.tok_pos(local_pos)
                events_emitted = False
                # Off first
                for p in sorted(off_events.get(step, [])):
                    if not events_emitted:
                        tokens.append(pos_tok)
                        events_emitted = True
                    tokens.append(self.tok_note_off(p))
                # On next
                for p in sorted(on_events.get(step, [])):
                    if not events_emitted:
                        tokens.append(pos_tok)
                        events_emitted = True
                    tokens.append(self.tok_note_on(p))
                    if self.config.include_velocity:
                        # Bucketize by 1..32
                        vbin = 16  # default mid velocity if unknown
                        tokens.append(self.tok_vel(vbin))

        tokens.append("<EOS>")
        # Map to ids
        return [self.vocab.get_id(t) for t in tokens]

    def decode(self, token_ids: List[int], fallback_bpm: int = 120) -> pm.PrettyMIDI:
        tokens = [self.vocab.get_token(i) for i in token_ids]
        # Parse header
        bpm = fallback_bpm
        program = 0
        for t in tokens[:16]:  # header window
            if t.startswith("BPM_"):
                try:
                    bpm = int(t.split("_", 1)[1])
                except Exception:
                    pass
            elif t.startswith("PROG_"):
                try:
                    program = int(t.split("_", 1)[1])
                except Exception:
                    pass

        step_sec = bpm_to_step_seconds(bpm)
        midi = pm.PrettyMIDI(initial_tempo=bpm)
        inst = pm.Instrument(program=program, is_drum=False)
        midi.instruments.append(inst)

        # State
        cur_bar = 1
        cur_pos = 0
        active: Dict[int, float] = {}

        def cur_step_index() -> int:
            base = 0 if cur_bar == 1 else STEPS_PER_BAR
            return min(base + cur_pos, TOTAL_STEPS - 1)

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "<EOS>":
                break
            if t.startswith("BAR_"):
                try:
                    cur_bar = int(t.split("_", 1)[1])
                except Exception:
                    cur_bar = 1
                cur_pos = 0
                i += 1
                continue
            if t.startswith("POS_"):
                try:
                    cur_pos = int(t.split("_", 1)[1])
                except Exception:
                    cur_pos = 0
                i += 1
                continue
            if t.startswith("NOTE_ON_"):
                pitch = int(t.split("_", 2)[2])
                step = cur_step_index()
                active[pitch] = step * step_sec
                i += 1
                # optional velocity token may follow; skip if present
                if i < len(tokens) and tokens[i].startswith("VEL_"):
                    i += 1
                continue
            if t.startswith("NOTE_OFF_"):
                pitch = int(t.split("_", 2)[2])
                if pitch in active:
                    start = active.pop(pitch)
                    end = max(start + step_sec, cur_step_index() * step_sec)
                    inst.notes.append(pm.Note(velocity=96, pitch=pitch, start=start, end=end))
                i += 1
                continue
            # skip other tokens
            i += 1

        # Clamp all notes within 2 bars
        max_time = (TOTAL_STEPS - 1) * step_sec
        for n in inst.notes:
            if n.end > max_time:
                n.end = max_time
            if n.end <= n.start:
                n.end = min(n.start + step_sec, max_time)
        return midi

    # ---------- Utilities ----------
    def save_vocab(self, path: str | Path):
        self.vocab.save(Path(path))

    def load_vocab(self, path: str | Path):
        self.vocab = Vocabulary.load(Path(path))


