"""Collect ALL identity pairs (input == expected) from train/val/test into one file
for manual inspection. identity is supposed to be CLEAN sentences the model must
leave untouched; in practice some are dirty (the sentence itself has spelling
errors), which both teaches the model to "keep wrong text" and makes the keep-rate
metric falsely low. Dump them all so a human can eyeball how dirty they are.

Output: one JSON line per identity pair, with a `split` field added.
Also prints a quick count + a sample.

Usage:
  python scripts/collect_identity.py corpus/splits/train.jsonl corpus/splits/val.jsonl corpus/splits/test.jsonl -o identity_all.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="jsonl splits to scan")
    p.add_argument("-o", "--out", default="identity_all.jsonl")
    args = p.parse_args()

    out_rows = []
    per_split = {}
    for fp in args.files:
        path = Path(fp)
        split = path.stem  # train / val / test
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("noise_type") == "identity":
                    out_rows.append({
                        "split": split,
                        "input": r.get("input", ""),
                        "expected": r.get("expected", ""),
                    })
                    n += 1
        per_split[split] = n

    with open(args.out, "w", encoding="utf-8") as w:
        for r in out_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"identity collected: {sum(per_split.values())}  -> {args.out}")
    for s, n in per_split.items():
        print(f"  {s:<8} {n}")
    print("\n  sample (first 20):")
    for r in out_rows[:20]:
        same = "==" if r["input"] == r["expected"] else "!="
        print(f"    [{r['split']}] {same}  {r['input'][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
