"""Per-noise_type P/R/F0.5 breakdown from a benchmark output dump.

The aggregate benchmark score collapses all noise types into one TP/FP/FN.
That hides WHERE the model is lazy. This regroups the per-sample dump by
noise_type and computes corpus-level P/R/F0.5 within each group, reusing the
exact same edit-counting (prf_counts) + fbeta as the benchmark -> numbers are
directly comparable to the headline F0.5.

Usage:
  python scripts/breakdown_by_noise.py output_bmd1905_vietnamese-correction-v2.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_common import prf_counts, fbeta  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/breakdown_by_noise.py <output_dump.jsonl>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # accumulate TP/FP/FN per noise_type + overall
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # [tp, fp, fn]
    cnt: dict[str, int] = defaultdict(int)
    for r in rows:
        nt = r.get("noise_type", "unknown")
        tp, fp, fn = prf_counts(r.get("input", ""), r.get("expected", ""), r.get("actual", ""))
        agg[nt][0] += tp
        agg[nt][1] += fp
        agg[nt][2] += fn
        agg["__ALL__"][0] += tp
        agg["__ALL__"][1] += fp
        agg["__ALL__"][2] += fn
        cnt[nt] += 1
        cnt["__ALL__"] += 1

    # report, sorted by F0.5 ascending (worst behavior first)
    print(f"\nModel: {rows[0].get('model','?')}   samples={cnt['__ALL__']}\n")
    header = f"{'noise_type':<20}{'n':>5}  {'TP':>5}{'FP':>5}{'FN':>5}   {'P':>6}{'R':>6}{'F0.5':>7}"
    print(header)
    print("-" * len(header))

    def line(nt: str) -> tuple[float, str]:
        tp, fp, fn = agg[nt]
        p, r, f = fbeta(tp, fp, fn)
        s = f"{nt:<20}{cnt[nt]:>5}  {tp:>5}{fp:>5}{fn:>5}   {p:>6.3f}{r:>6.3f}{f:>7.3f}"
        return f, s

    per_type = [nt for nt in agg if nt != "__ALL__"]
    for nt in sorted(per_type, key=lambda x: line(x)[0]):
        print(line(nt)[1])
    print("-" * len(header))
    print(line("__ALL__")[1])

    print("\nReading: low R = lazy (skips edits) | low P = reckless (wrong edits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
