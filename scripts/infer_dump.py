"""Run a local fine-tuned seq2seq model over a test set and dump per-sample outputs.

Produces the same JSONL schema the benchmark dump uses
({model, input, expected, actual, noise_type, ...}) so breakdown_by_noise.py can
consume it directly and compare against the raw-v2 baseline.

CPU-friendly: supports a stratified subsample (full 49k on CPU is very slow).

Usage:
  python scripts/infer_dump.py checkpoint_model/checkpoint-3000 \
      corpus/splits/test.jsonl output_modelA.jsonl --sample 3000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Proportional sample by noise_type so the subset mirrors the full distribution."""
    import random
    rng = random.Random(seed)
    by_nt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_nt[r.get("noise_type", "unknown")].append(r)
    total = len(rows)
    out: list[dict] = []
    for nt, group in by_nt.items():
        rng.shuffle(group)
        quota = max(1, round(n * len(group) / total))
        out.extend(group[:quota])
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    p = argparse.ArgumentParser(description="Infer a local seq2seq model over a test set, dump outputs.")
    p.add_argument("model_dir", help="Path to the fine-tuned model folder")
    p.add_argument("test_jsonl", help="Path to test.jsonl (input, expected, noise_type)")
    p.add_argument("output_jsonl", help="Where to write per-sample outputs")
    p.add_argument("--sample", type=int, default=0, help="Stratified subsample size (0=full)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = load_jsonl(Path(args.test_jsonl))
    if args.sample and len(rows) > args.sample:
        rows = stratified_sample(rows, args.sample, args.seed)
        dist = Counter(r.get("noise_type", "unknown") for r in rows)
        print(f"Stratified subsample: {len(rows)} ({dict(dist)})", flush=True)
    else:
        print(f"Using full set: {len(rows)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model_dir} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    model_name = Path(args.model_dir).name
    out_f = open(args.output_jsonl, "w", encoding="utf-8")
    t0 = time.time()
    done = 0
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [r["input"] for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=args.max_len, num_beams=args.num_beams,
                                  early_stopping=True)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for r, act in zip(batch, decoded):
            rec = {
                "model": model_name,
                "input": r["input"],
                "expected": r["expected"],
                "actual": act,
                "noise_type": r.get("noise_type", "unknown"),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += len(batch)
        if (i // args.batch_size) % 5 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            eta = (len(rows) - done) / max(rate, 1e-6)
            print(f"  {done}/{len(rows)}  ({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)
    out_f.close()
    print(f"Done -> {args.output_jsonl}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
