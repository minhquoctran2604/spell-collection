"""Benchmark a seq2seq corrector on a GENERAL-domain spell-correction test set
(Viwiki / VSEC), reusing the same F0.5 edit metric as the YHCT benchmark.

Reads a jsonl of {input, expected[, noise_type]}, runs the model, and reports
corpus-level precision / recall / F0.5 (beta=0.5) plus exact-match and identity
keep-rate, using benchmark_common (identical algorithm to the YHCT headline).

Dumps per-sentence outputs for eyeballing.

Usage:
  python scripts/benchmark_general.py bmd1905/vietnamese-correction-v2 \
      baochi/viwiki_pairs.jsonl --out-dir baochi --tag v2_viwiki
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import unicodedata


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


# reuse the metric primitives from benchmark_common (same F0.5 as YHCT headline)
from benchmark_common import prf_counts, fbeta  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("test_jsonl")
    p.add_argument("--out-dir", default="baochi")
    p.add_argument("--tag", default="bench")
    p.add_argument("--max", type=int, default=0, help="cap samples (0=all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8") if l.strip()]
    if args.max:
        rows = rows[: args.max]
    print(f"test samples: {len(rows)}  model: {args.model}", flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    print(f"loaded on {device}", flush=True)

    tp = fp = fn = 0
    exact = 0
    dump = []
    bs = args.batch_size
    t0 = time.time()
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        inp = [r["input"] for r in batch]
        enc = tok(inp, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=args.max_len, num_beams=args.num_beams)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for r, out in zip(batch, dec):
            a, b, c = prf_counts(r["input"], r["expected"], out)
            tp += a; fp += b; fn += c
            if nfc(out) == nfc(r["expected"]):
                exact += 1
            dump.append({"input": r["input"], "expected": r["expected"], "output": out})
        if (i // bs) % 5 == 0:
            print(f"  {i+len(batch)}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    n = len(rows)
    prec, rec, f05 = fbeta(tp, fp, fn, beta=0.5)
    em = exact / n if n else 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"bench_{args.tag}_outputs.jsonl"
    with open(dump_path, "w", encoding="utf-8") as w:
        for d in dump:
            w.write(json.dumps(d, ensure_ascii=False) + "\n")
    report = {
        "model": args.model,
        "test": args.test_jsonl,
        "n": n,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f05": round(f05, 4),
        "exact_match": round(em, 4),
    }
    rep_path = out_dir / f"bench_{args.tag}_report.json"
    with open(rep_path, "w", encoding="utf-8") as w:
        json.dump(report, w, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"GENERAL BENCHMARK  ({args.tag})")
    print("=" * 60)
    print(f"  model        : {args.model}")
    print(f"  samples      : {n}")
    print(f"  precision    : {prec:.4f}")
    print(f"  recall       : {rec:.4f}")
    print(f"  F0.5         : {f05:.4f}")
    print(f"  exact match  : {em:.4f}  ({exact}/{n})")
    print(f"  tp/fp/fn     : {tp}/{fp}/{fn}")
    print(f"\n  report -> {rep_path}")
    print(f"  outputs-> {dump_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
