"""MEASURE 2 — does the training data teach the model to DELETE words?

Premise: gen_variants makes noisy INPUT from clean EXPECTED by corrupting spelling
(drop diacritics, wrong tone, swap chars, remove spaces). NONE of these noise types
should DROP a whole word. So for a clean corpus, word-count(input) should equal
word-count(expected) — except for space-noise (missing_space / nospace), where
removing/adding spaces legitimately changes the whitespace-token count.

If we find many pairs where `expected` has FEWER words than `input` (outside the
space-noise types), that means the gold itself is short a word — i.e. the data is
teaching a long->short (deletion) mapping. That would make "train harder to stop
deletion" futile until the data is filtered.

This is the COUNTERPART to measure_corpus_dirtiness.py:
  - dirtiness  = does gold have spelling errors?     (char-level)
  - this (M2)  = does gold drop whole words vs input? (token-count level)

Method (deterministic, no model, fast over ~1M lines):
  for each pair: wi = #words(input), we = #words(expected)
    delta = wi - we
    bucket by sign, by noise_type; collect examples where we < wi (expected shorter)

Usage:
  python scripts/measure_word_deletion.py corpus/training.jsonl
  python scripts/measure_word_deletion.py corpus/splits/train.jsonl --examples 30
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def nwords(s: str) -> int:
    return len(nfc(s).split())


# noise types where whitespace-token count legitimately changes
SPACE_NOISE = {"missing_space", "nospace"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--examples", type=int, default=25,
                   help="how many `expected shorter` examples to print")
    args = p.parse_args()

    path = Path(args.jsonl)
    total = 0
    # delta = wi - we  ; delta>0 means expected has FEWER words (suspicious)
    exp_shorter = 0          # we < wi
    exp_longer = 0           # we > wi
    same = 0                 # we == wi
    by_nt_total = Counter()
    by_nt_shorter = Counter()
    # among expected-shorter, split space-noise vs NON-space-noise (the real concern)
    shorter_nonspace = 0
    shorter_space = 0
    delta_hist = Counter()   # delta value -> count (for expected-shorter only)
    examples = []            # (noise_type, delta, input, expected)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            inp = r.get("input", "")
            exp = r.get("expected", "")
            nt = r.get("noise_type", "unknown")
            total += 1
            by_nt_total[nt] += 1
            wi, we = nwords(inp), nwords(exp)
            delta = wi - we
            if delta > 0:
                exp_shorter += 1
                by_nt_shorter[nt] += 1
                delta_hist[delta] += 1
                if nt in SPACE_NOISE:
                    shorter_space += 1
                else:
                    shorter_nonspace += 1
                    if len(examples) < args.examples:
                        examples.append((nt, delta, inp, exp))
            elif delta < 0:
                exp_longer += 1
            else:
                same += 1

    print("=" * 70)
    print(f"WORD-DELETION CHECK  (file={path.name}, pairs={total})")
    print("=" * 70)
    print(f"  expected == input words : {same:>8}  ({same/total*100:.2f}%)")
    print(f"  expected  > input words : {exp_longer:>8}  ({exp_longer/total*100:.2f}%)  (input merged words / space-noise)")
    print(f"  expected  < input words : {exp_shorter:>8}  ({exp_shorter/total*100:.2f}%)  <-- gold dropped a word vs input")
    print()
    print(f"  Of the {exp_shorter} 'expected shorter' pairs:")
    print(f"    in SPACE-noise types ({'/'.join(sorted(SPACE_NOISE))}): {shorter_space:>8}  (EXPECTED, legit)")
    print(f"    in OTHER noise types (THE CONCERN)               : {shorter_nonspace:>8}  ({shorter_nonspace/total*100:.3f}% of all pairs)")
    print()
    print(f"  delta (#words dropped) histogram among expected-shorter:")
    for d in sorted(delta_hist):
        print(f"    -{d:<3} words : {delta_hist[d]:>7}")
    print()
    print(f"  expected-shorter rate per noise_type (shorter / total):")
    for nt in sorted(by_nt_total):
        s, t = by_nt_shorter[nt], by_nt_total[nt]
        flag = "" if nt in SPACE_NOISE else "  <-- concern" if s else ""
        print(f"    {nt:<18} {s:>7}/{t:<8} ({s/t*100:5.2f}%){flag}")
    print()
    print(f"  examples of NON-space-noise expected-shorter (gold genuinely drops a word):")
    if not examples:
        print("    (none — clean: no word-deletion outside space-noise)")
    for nt, d, inp, exp in examples:
        print(f"    [{nt}] -{d}w")
        print(f"      in : {inp[:100]}")
        print(f"      exp: {exp[:100]}")
    print()
    print("  READING: if NON-space-noise expected-shorter ~ 0%, data is CLEAN on word count")
    print("  -> deletion is NOT taught by data; train-harder to override v2 is viable.")
    print("  If it is sizeable, the gold itself drops words -> FILTER those pairs first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
