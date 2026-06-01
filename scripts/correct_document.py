"""Run a fine-tuned corrector over a REAL document (txt or docx) and show before/after.

This is the real-world smoke test: synthetic F0.5 means nothing if the model fails
on actual OCR output. Feed a real OCR'd YHCT document, correct it sentence by
sentence, and dump an aligned before/after so a human can eyeball real errors.

Sentence segmentation: simple — split on newlines, then on sentence-final
punctuation. Good enough for eyeballing; not the training segmenter.

Reads .txt directly; reads .docx via python-docx (paragraph text).

Usage:
  python scripts/correct_document.py checkpoint_model/checkpoint-3000 input.docx out.txt
  python scripts/correct_document.py checkpoint_model/checkpoint-3000 input.txt  out.txt --max 200
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_document(path: Path) -> list[str]:
    """Return a list of paragraph/line strings from txt or docx."""
    if path.suffix.lower() == ".docx":
        try:
            from docx import Document  # python-docx
        except ImportError:
            print("Need python-docx: pip install python-docx", file=sys.stderr)
            raise
        doc = Document(str(path))
        return [p.text for p in doc.paragraphs if p.text.strip()]
    # txt
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def to_sentences(paragraphs: list[str]) -> list[str]:
    sents: list[str] = []
    for p in paragraphs:
        parts = _SENT_SPLIT.split(p.strip())
        for s in parts:
            s = s.strip()
            if s:
                sents.append(s)
    return sents


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("input_doc", help="real OCR document (.txt or .docx)")
    p.add_argument("output_txt", help="aligned before/after output")
    p.add_argument("--max", type=int, default=0, help="cap number of sentences (0=all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--num-beams", type=int, default=5)
    args = p.parse_args()

    paras = read_document(Path(args.input_doc))
    sents = to_sentences(paras)
    if args.max:
        sents = sents[: args.max]
    print(f"document -> {len(paras)} paragraphs -> {len(sents)} sentences", flush=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()
    print(f"model loaded on {device}", flush=True)

    out_f = open(args.output_txt, "w", encoding="utf-8")
    changed = 0
    bs = args.batch_size
    t0 = time.time()
    for i in range(0, len(sents), bs):
        batch = sents[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_length=args.max_len, num_beams=args.num_beams)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for src, out in zip(batch, dec):
            mark = "" if src == out else "  <-- CHANGED"
            if src != out:
                changed += 1
            out_f.write(f"IN : {src}\n")
            out_f.write(f"OUT: {out}{mark}\n\n")
        if (i // bs) % 5 == 0:
            print(f"  {i+len(batch)}/{len(sents)} ({time.time()-t0:.0f}s)", flush=True)
    out_f.close()
    print(f"\nDone -> {args.output_txt}")
    print(f"  {changed}/{len(sents)} sentences changed ({changed/max(len(sents),1)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
