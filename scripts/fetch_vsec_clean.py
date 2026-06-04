"""Download VSEC (bmd1905... no — nguyenthanhasia/vsec) and extract the CLEAN
sentences (corrected_text) into a plain .txt (one sentence per line), ready to
feed into the existing gen_variants.py pipeline.

We only take corrected_text (the clean side). The real wrong->correct pairs are
left untouched / handled separately; here we just harvest clean sentences as raw
material for synthetic noise generation, exactly like sentences_clean.txt.

Dedup + drop too-short/empty lines.

Usage:
  python scripts/fetch_vsec_clean.py --out baochi/vsec_clean.txt
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nguyenthanhasia/vsec-vietnamese-spell-correction")
    p.add_argument("--split", default="train")
    p.add_argument("--field", default="corrected_text", help="clean-text column")
    p.add_argument("--out", default="baochi/vsec_clean.txt")
    p.add_argument("--min-len", type=int, default=10, help="min chars to keep a line")
    args = p.parse_args()

    from datasets import load_dataset
    print(f"loading {args.dataset} [{args.split}] ...", flush=True)
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  rows: {len(ds)}  columns: {ds.column_names}", flush=True)
    if args.field not in ds.column_names:
        print(f"ERROR: field '{args.field}' not in columns {ds.column_names}", file=sys.stderr)
        return 1

    seen = set()
    lines = []
    for s in ds[args.field]:
        s = nfc(s).strip()
        if len(s) < args.min_len:
            continue
        if s in seen:
            continue
        seen.add(s)
        # one line per sentence; collapse any embedded newlines
        lines.append(" ".join(s.split("\n")).strip())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} clean sentences -> {out}")
    print("\n  sample:")
    for ln in lines[:8]:
        print("   ", ln[:100])
    return 0


if __name__ == "__main__":
    sys.exit(main())
