"""Assemble Viwiki-spelling docs into (wrong -> correct) sentence pairs.

Input (baochi/spelling_test.json), JSON-lines, each doc:
  { "_id", "text": "<full doc, ALREADY CONTAINS the spelling errors>",
    "mistakes": [ { "text": "<wrong word>", "start_offset": <int-or-str>,
                    "suggest": ["<correct word>", ...] }, ... ] }

`text` is the ERRONEOUS document. We build the CORRECT document by replacing each
mistake span with its first suggestion. CRITICAL: apply replacements from the
LAST offset to the FIRST (descending) so earlier offsets stay valid when a
replacement changes length (e.g. "là"->"làm").

Then split both wrong/correct docs into sentences (same splitter, 1:1 because
replacements never change sentence count) and emit one pair per sentence that
actually changed (the corrector's job is those). Clean (unchanged) sentences are
emitted as identity pairs only if --keep-identity.

Output (baochi/viwiki_pairs.jsonl): {"input","expected","noise_type":"viwiki"}.

Usage:
  python scripts/build_viwiki_pairs.py baochi/spelling_test.json -o baochi/viwiki_pairs.jsonl
  python scripts/build_viwiki_pairs.py baochi/spelling_test.json -o baochi/viwiki_pairs.jsonl --keep-identity
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")


def split_sents(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def apply_fixes(text: str, mistakes: list[dict]) -> tuple[str, int, int]:
    """Return (corrected_text, applied, skipped). Replace each mistake span with
    suggest[0], descending by offset so length changes don't shift later spans."""
    applied = skipped = 0
    ms = []
    for m in mistakes:
        try:
            off = int(m["start_offset"])
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        wrong = str(m.get("text", ""))
        sug = m.get("suggest") or []
        if not wrong or not sug:
            skipped += 1
            continue
        ms.append((off, wrong, str(sug[0])))
    # descending by offset
    ms.sort(key=lambda t: t[0], reverse=True)
    for off, wrong, right in ms:
        # sanity: the span at off must equal the wrong word; else skip (offset drift / bad label)
        if text[off:off + len(wrong)] == wrong:
            text = text[:off] + right + text[off + len(wrong):]
            applied += 1
        else:
            skipped += 1
    return text, applied, skipped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("-o", "--out", default="baochi/viwiki_pairs.jsonl")
    p.add_argument("--keep-identity", action="store_true",
                   help="also emit unchanged (clean) sentences as identity pairs")
    args = p.parse_args()

    docs = [json.loads(l) for l in open(args.infile, encoding="utf-8") if l.strip()]
    out_pairs = []
    tot_applied = tot_skipped = 0
    docs_with_offset_issue = 0

    for d in docs:
        wrong_doc = nfc(d.get("text", ""))
        fixed_doc, ap, sk = apply_fixes(wrong_doc, d.get("mistakes", []))
        fixed_doc = nfc(fixed_doc)
        tot_applied += ap
        tot_skipped += sk
        if sk:
            docs_with_offset_issue += 1

        w_sents = split_sents(wrong_doc)
        f_sents = split_sents(fixed_doc)
        if len(w_sents) != len(f_sents):
            # sentence count desync (rare; replacement spanned a boundary) -> skip doc to stay safe
            continue
        for ws, fs in zip(w_sents, f_sents):
            if ws != fs:
                out_pairs.append({"input": ws, "expected": fs, "noise_type": "viwiki"})
            elif args.keep_identity:
                out_pairs.append({"input": ws, "expected": fs, "noise_type": "identity"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for r in out_pairs:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    changed = sum(1 for r in out_pairs if r["noise_type"] == "viwiki")
    ident = sum(1 for r in out_pairs if r["noise_type"] == "identity")
    print("=" * 60)
    print(f"VIWIKI PAIR BUILD")
    print("=" * 60)
    print(f"  docs                 : {len(docs)}")
    print(f"  fixes applied        : {tot_applied}")
    print(f"  fixes skipped (drift): {tot_skipped}  (docs affected: {docs_with_offset_issue})")
    print(f"  changed-sentence pairs: {changed}")
    if args.keep_identity:
        print(f"  identity pairs       : {ident}")
    print(f"  total written        : {len(out_pairs)}  -> {args.out}")
    print(f"\n  sample pairs:")
    for r in [x for x in out_pairs if x['noise_type'] == 'viwiki'][:6]:
        print(f"    IN : {r['input'][:90]}")
        print(f"    EXP: {r['expected'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
