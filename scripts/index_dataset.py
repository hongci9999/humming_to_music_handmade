import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


LABEL_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "label"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "source"


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _map_scale_str(scale: str) -> str:
    if not scale:
        return ""
    s = scale.strip()
    # Normalize unicode/spacing and case
    s = s.replace("♭", "b").replace("♯", "#")
    parts = s.split()
    if len(parts) == 2:
        root, mode = parts
        return f"{root.upper()} {mode.lower()}"
    return s


def find_source_candidate(src_file_name: str) -> str:
    """Best-effort match of WAV under dataset/source zips/folders.

    The repository may store audio inside structured folders like
    TS_XX.*.zip/. We return a relative path string if a likely match is found;
    otherwise, empty string.
    """
    if not src_file_name:
        return ""

    # Walk only first-level category folders to keep it fast
    for cat_dir in sorted(SOURCE_ROOT.glob("TS_*")):
        # Search both directly inside and recursively (zip views may appear as dirs)
        for p in cat_dir.rglob(src_file_name):
            return str(p.relative_to(SOURCE_ROOT))
    return ""


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    json_files = sorted(LABEL_ROOT.rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        ds = data.get("dataSet", {})
        loop_index = ds.get("loopIndex", "")
        bar_count = ds.get("barCount", None)
        beat_count = _safe_get(ds, ["loopInfo", "beatCount"], None)
        bpm = _safe_get(ds, ["loopInfo", "bpm"], None)
        scale_raw = _safe_get(ds, ["loopInfo", "scale"], "")
        scale = _map_scale_str(scale_raw)
        instrument_type = ds.get("InstrumentType", "")
        instrument_name = _safe_get(ds, ["loopInfo", "InstrumentName"], "")
        genre = _safe_get(ds, ["loopInfo", "genre"], "")
        style = _safe_get(ds, ["loopInfo", "musicStyle"], "")
        form = _safe_get(ds, ["loopInfo", "songForm"], "")
        is_main = ds.get("isMain", "")
        play_time_ms = ds.get("playTime", None)
        src_file_name = ds.get("srcFileName", "")

        wav_rel = find_source_candidate(src_file_name)

        rows.append(
            {
                "label_json": str(jf.relative_to(LABEL_ROOT)),
                "wav_path": wav_rel,
                "wav_exists": bool(wav_rel),
                "instrument_type": instrument_type,
                "instrument_name": instrument_name,
                "genre": genre,
                "style": style,
                "form": form,
                "bpm": bpm,
                "scale": scale,
                "bar_count": bar_count,
                "beat_count": beat_count,
                "play_time_ms": play_time_ms,
                "loop_index": loop_index,
                "is_main": is_main,
            }
        )
    return rows


def main(output: str):
    rows = collect_rows()
    df = pd.DataFrame(rows)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        # default to csv
        df.to_csv(out_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "dataset_index.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args.output)


