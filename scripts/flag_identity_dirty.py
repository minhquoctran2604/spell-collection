"""Flag SUSPECT-dirty identity expected strings via cheap heuristics (no model).

identity pair = input == expected, meant to be a CLEAN sentence. We only need to
inspect `expected`: if it's clean, the pair is fine. We flag strings that show
TELL-TALE OCR / typo signatures so a human can eyeball them:

  - lowercase letter stuck inside an UPPERCASE word   (PHAcH, ORGANiC)
  - a letter directly adjacent to a digit inside a token (mS, lO, l0)
  - leading stray punctuation  (' Quyết, - xxx)
  - isolated single non-vowel letter token (OCR fragment)
  - repeated word back-to-back  ("dân dân", "các các")
  - non-Vietnamese / non-ASCII oddball chars
  - very long run with no space (merge) or weird mixed-case mid-word

These are HEURISTICS — they over- and under-catch. The point is to surface
candidates, not to auto-delete. Clean YHCT Hán-Việt terms may trip some rules;
the human confirms.

Usage:
  python scripts/flag_identity_dirty.py identity_all.jsonl --out identity_suspect.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


VI_LOWER = "a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
VI_UPPER = "A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
VI_CHARS = VI_LOWER + VI_UPPER

WORD_RE = re.compile(rf"[{VI_CHARS}]+")
LEAD_PUNCT = re.compile(r"^\s*['\"`^~*•·,;:.)\]}\-]")
LETTER_DIGIT = re.compile(rf"[{VI_CHARS}][0-9]|[0-9][{VI_CHARS}]")
# lowercase inside an otherwise-uppercase token: e.g. PHAcH
MIXED_CASE = re.compile(rf"[{VI_UPPER}]{{2,}}[{VI_LOWER}][{VI_UPPER}]|[{VI_UPPER}][{VI_LOWER}][{VI_UPPER}]{{2,}}")


def repeated_word(s: str) -> bool:
    toks = WORD_RE.findall(nfc(s).lower())
    return any(a == b and len(a) > 1 for a, b in zip(toks, toks[1:]))


def isolated_consonant(s: str) -> str | None:
    # single-letter token that's not a real vi 1-letter word (a, à, ô, ơ, y, ...)
    ok1 = set(
        "aàáảãạ"
        "eèéẻẽẹ"
        "êềếểễệ"
        "iìíỉĩị"
        "oòóỏõọ"
        "ôồốổỗộ"
        "ơờớởỡợ"
        "uùúủũụ"
        "ưừứửữự"
        "yỳýỷỹỵ"
    )
    for tok in WORD_RE.findall(nfc(s)):
        if len(tok) == 1 and tok.lower() not in ok1:
            return tok
    return None


def weird_chars(s: str) -> str | None:
    for ch in nfc(s):
        o = ord(ch)
        # allow basic latin, vi diacritics block, common punct, digits, spaces
        if ch.isspace():
            continue
        if ch in ".,;:!?()[]{}\"'`-–—…/%°+×=<>$&@#*•·²³":
            continue
        if ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            # letter: ok if latin/vi; flag CJK, cyrillic etc.
            try:
                name = unicodedata.name(ch)
            except ValueError:
                return ch
            if "LATIN" not in name:
                return ch
            continue
        # any other symbol category -> suspicious
        return ch
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--out", default="identity_suspect.jsonl")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    reasons = Counter()
    suspects = []

    for r in rows:
        exp = nfc(r.get("expected", ""))
        flags = []
        if LEAD_PUNCT.search(exp):
            flags.append("lead_punct")
        if LETTER_DIGIT.search(exp):
            flags.append("letter_digit")
        if MIXED_CASE.search(exp):
            flags.append("mixed_case")
        if repeated_word(exp):
            flags.append("repeated_word")
        ic = isolated_consonant(exp)
        if ic:
            flags.append(f"lone_letter:{ic}")
        wc = weird_chars(exp)
        if wc:
            flags.append(f"weird_char")
        if flags:
            for f in flags:
                reasons[f.split(":")[0]] += 1
            suspects.append({**r, "flags": flags})

    with open(args.out, "w", encoding="utf-8") as w:
        for s in suspects:
            w.write(json.dumps(s, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"identity total   : {n}")
    print(f"flagged suspect  : {len(suspects)}  ({len(suspects)/n*100:.1f}%)")
    print(f"\n  by reason:")
    for k, v in reasons.most_common():
        print(f"    {k:<16} {v}")
    print(f"\n  -> {args.out}  (human-confirm; heuristics over/under-catch)")
    print(f"\n  examples per reason:")
    shown = Counter()
    for s in suspects:
        key = s["flags"][0].split(":")[0]
        if shown[key] < 5:
            shown[key] += 1
            print(f"    [{key}] {s['expected'][:95]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
