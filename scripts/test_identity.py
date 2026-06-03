"""Measure over-correction behavior of a model on identity samples only.

identity = input already correct (input == expected). The model SHOULD leave it
unchanged. The headline F0.5 is meaningless here (no gold edits -> always 0). The
right metric is the KEEP RATE: fraction of clean sentences returned verbatim, plus
how many spurious edits (FP) the model introduces on the rest.

This isolates the over-correction question with a large identity sample (the
in-training eval only saw ~20 identity sentences -> too noisy to decide weights).

Usage:
  python scripts/test_identity.py checkpoint_model/checkpoint-3000 corpus/splits/test.jsonl
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def n_edits(a: str, b: str) -> int:
    """Number of non-equal char-span ops between a and b (spurious edits if a is clean)."""
    sm = SequenceMatcher(None, nfc(a), nfc(b), autojunk=False)
    return sum(1 for tag, *_ in sm.get_opcodes() if tag != "equal")


import re as _re
# All punctuation / quotes / brackets we treat as NOISE (differences here are NOT
# real over-corrections — dropping a period, a quote, a trailing ! is harmless).
_PUNCT = _re.compile(r"""[\s.,;:!?()\[\]{}"'`“”‘’«»…\-–—/\\|*•·]+""")


def norm_punct(s: str) -> str:
    """Lowercase + strip ALL punctuation/quotes/whitespace, so two strings that
    differ ONLY in punctuation compare equal. Real letter/word changes survive."""
    return _PUNCT.sub("", nfc(s).lower())


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python scripts/test_identity.py <model_dir> <test.jsonl> [--max N] [--beams K]", file=sys.stderr)
        return 1
    model_dir = sys.argv[1]
    test_path = sys.argv[2]
    max_n = 0
    beams = 1  # greedy: identity needs verbatim copy, beam search not needed -> faster on CPU
    if "--max" in sys.argv:
        max_n = int(sys.argv[sys.argv.index("--max") + 1])
    if "--beams" in sys.argv:
        beams = int(sys.argv[sys.argv.index("--beams") + 1])

    # collect identity rows
    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("noise_type") == "identity":
                rows.append(r)
    if max_n:
        rows = rows[:max_n]
    print(f"identity samples: {len(rows)}  (beams={beams})", flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    print(f"model loaded on {device}", flush=True)

    kept = 0          # output == input (verbatim, perfect)
    kept_soft = 0     # output == input IGNORING punctuation differences
    changed = 0       # output != input (over-corrected)
    punct_only = 0    # differs from input ONLY by punctuation (harmless)
    total_fp = 0      # total spurious edits across all samples
    examples = []     # a few REAL (non-punct) over-correction examples to eyeball
    bs = 16
    t0 = time.time()
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        texts = [r["input"] for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=256, num_beams=beams, early_stopping=True)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for r, out in zip(batch, dec):
            verbatim = nfc(out) == nfc(r["input"])
            soft_equal = norm_punct(out) == norm_punct(r["input"])
            if verbatim:
                kept += 1
                kept_soft += 1
            elif soft_equal:
                # differs ONLY by punctuation -> harmless, count as soft-kept
                kept_soft += 1
                punct_only += 1
            else:
                changed += 1
                e = n_edits(r["input"], out)
                total_fp += e
                if len(examples) < 15:
                    examples.append((r["input"], out, e))
        if (i // bs) % 5 == 0:
            print(f"  {i+len(batch)}/{len(rows)}  ({(time.time()-t0):.0f}s)", flush=True)

    n = len(rows)
    print("\n" + "=" * 60)
    print(f"IDENTITY OVER-CORRECTION REPORT  (model={Path(model_dir).name})")
    print("=" * 60)
    print(f"  total identity        : {n}")
    print(f"  kept verbatim (strict): {kept}  ({kept/n*100:.1f}%)")
    print(f"  punct-only diff (ok)  : {punct_only}  ({punct_only/n*100:.1f}%)")
    print(f"  REAL over-correct     : {changed}  ({changed/n*100:.1f}%)  <- letters/words changed")
    print(f"  total spurious edits  : {total_fp}  (avg {total_fp/max(n,1):.2f}/sample)")
    print(f"  KEEP RATE (strict)    : {kept/n:.3f}   (counts punct diffs as bad)")
    print(f"  KEEP RATE (soft)      : {kept_soft/n:.3f}   (ignores punct diffs)  <-- FAIR metric")
    print("\n  sample REAL over-corrections (input -> output, #edits):")
    for inp, out, e in examples:
        print(f"    [{e}] {inp}")
        print(f"        -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
