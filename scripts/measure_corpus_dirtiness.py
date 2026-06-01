"""Estimate how DIRTY the gold `expected` labels are, using a correction model as judge.

Premise: `expected` is supposed to be clean ground truth. The syllable-validator
filter only removes STRUCTURALLY impossible OCR garbage ("hgười"); it lets through
subtle spelling errors that are still valid syllables ("Qúa"->"Quá", "Đài"->"Đại",
"túy"->"tủy"). If `expected` still contains such errors, every metric computed
against it is biased (the model gets penalized for correctly fixing a dirty gold).

Method: feed each `expected` string to a strong correction model. If the model
CHANGES it, the expected is suspected dirty. Report the fraction changed, overall
and per noise_type, plus the average edit count and examples.

CAVEAT: this is a LOWER BOUND on dirtiness — the judge model has imperfect recall
(misses some errors), so true dirtiness is >= measured. Also the judge may
over-correct (false alarm), so treat the number as an order-of-magnitude estimate,
and READ the examples to confirm they are real errors.

Usage:
  python scripts/measure_corpus_dirtiness.py corpus/splits/test.jsonl \
      --model bmd1905/vietnamese-correction-v2 --sample 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def n_edits(a: str, b: str) -> int:
    sm = SequenceMatcher(None, nfc(a), nfc(b), autojunk=False)
    return sum(1 for tag, *_ in sm.get_opcodes() if tag != "equal")


def stratified(rows, n, seed=42):
    import random
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r.get("noise_type", "unknown")].append(r)
    tot = len(rows)
    out = []
    for nt, g in by.items():
        rng.shuffle(g)
        out.extend(g[: max(1, round(n * len(g) / tot))])
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("test_jsonl")
    p.add_argument("--model", default="bmd1905/vietnamese-correction-v2")
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--beams", type=int, default=5)
    p.add_argument("--dump", default="dirty_suspects.jsonl",
                   help="Write ALL suspected-dirty pairs here for manual review")
    args = p.parse_args()

    rows = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.sample and len(rows) > args.sample:
        rows = stratified(rows, args.sample)
    print(f"judging {len(rows)} expected labels with {args.model} (beams={args.beams})", flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    print(f"loaded on {device}", flush=True)

    by_nt_total = Counter()
    by_nt_dirty = Counter()
    total_edits = 0
    examples = []
    dump_f = open(args.dump, "w", encoding="utf-8")
    bs = 16
    t0 = time.time()
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        exps = [r["expected"] for r in batch]
        enc = tok(exps, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=256, num_beams=args.beams)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for r, out in zip(batch, dec):
            nt = r.get("noise_type", "unknown")
            by_nt_total[nt] += 1
            if nfc(out) != nfc(r["expected"]):
                by_nt_dirty[nt] += 1
                e = n_edits(r["expected"], out)
                total_edits += e
                # dump EVERY suspect for manual review (the judge model is noisy;
                # the human decides real-dirty vs judge-false-alarm)
                dump_f.write(json.dumps({
                    "noise_type": nt,
                    "expected": r["expected"],
                    "judge_fixed": out,
                    "n_edits": e,
                }, ensure_ascii=False) + "\n")
                if len(examples) < 15:
                    examples.append((r["expected"], out, e))
        if (i // bs) % 10 == 0:
            print(f"  {i+len(batch)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    dump_f.close()

    n = len(rows)
    dirty = sum(by_nt_dirty.values())
    print("\n" + "=" * 64)
    print(f"CORPUS DIRTINESS (judge={args.model}) — LOWER BOUND")
    print("=" * 64)
    print(f"  expected judged     : {n}")
    print(f"  suspected dirty     : {dirty}  ({dirty/n*100:.1f}%)")
    print(f"  avg edits on dirty  : {total_edits/max(dirty,1):.2f}")
    print(f"\n  per noise_type (dirty / total):")
    for nt in sorted(by_nt_total):
        t = by_nt_total[nt]; d = by_nt_dirty[nt]
        print(f"    {nt:<18} {d:>4}/{t:<4} ({d/t*100:.1f}%)")
    print(f"\n  examples (expected -> judge's correction, #edits):")
    for exp, out, e in examples:
        print(f"    [{e}] {exp[:90]}")
        print(f"        -> {out[:90]}")
    print(f"\n  NOTE: lower bound (judge recall imperfect). READ examples — confirm real errors vs judge false-alarm.")
    print(f"  ALL {dirty} suspects dumped -> {args.dump}  (review manually to separate real-dirty from judge over-correction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
