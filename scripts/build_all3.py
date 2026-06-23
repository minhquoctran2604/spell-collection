"""Build train / val / test — ALL balanced to the same target, verified by vn_errtype,
disjoint sentences (anti-leak), each with typing + OCR + identity.

Sentence partition (no overlap): corpus shuffled once, split into 3 disjoint pools
by index range, so train/val/test never share a source sentence.

Each split:
  TYPING errors  : build_balanced.build_typing_pool (verified vn_errtype distribution)
  OCR errors     : gen_ocr corruptors (8 mechanisms, own target)
  identity       : clean keep sentences
  class ratio    : TYPING ~41% / OCR ~41% / identity ~18%

Usage:
  python scripts/build_all3.py --train-err 9300 --val-err 1800 --test-err 1200
"""
import argparse, json, re, sys, io, os, random
from pathlib import Path
import importlib.util
def _load(name):
    sp=importlib.util.spec_from_file_location(name, str(Path(__file__).parent/f"{name}.py"))
    m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
ve=_load("vn_errtype"); bb=_load("build_balanced"); go=_load("gen_ocr")
# imported modules wrap+close sys.stdout; reopen fresh from fd 1
sys.stdout = io.TextIOWrapper(io.FileIO(os.dup(1), "w"), encoding="utf-8")
nfc=ve.nfc

def load_corpus(paths):
    seen=set(); out=[]
    for p in paths:
        if not Path(p).exists(): continue
        for l in open(p,encoding="utf-8"):
            s=nfc(l.strip())
            if 20<=len(s)<=200 and s not in seen:
                seen.add(s); out.append(s)
    return out

def ocr_pool(clean, n_err, leak, seed, multi_frac=0.30):
    """Reuse gen_ocr corruptors. A multi_frac of rows get a SECOND word-corruptor pass
    so OCR also carries multi-error sentences (was 0% before). count is re-measured by
    vn_errtype (single/multi), not assumed."""
    rng=go.LCG(seed); rows=[]; counts={}; pi=0
    sents=[s for s in clean if s not in leak]
    random.Random(seed).shuffle(sents)
    word_types=set(go.CORRUPT); wt=list(word_types)
    for ntype,share in go.TARGET.items():
        want=round(n_err*share); made=0; tries=0
        while made<want and pi<len(sents) and tries<want*10:
            s=sents[pi]; pi+=1; tries+=1
            if ntype in word_types: v=go.apply_word_corruptor(s, go.CORRUPT[ntype], rng)
            elif ntype=="ocr_punct": v=go.c_punct(s, rng)
            else: v=go.c_split_merge(s, rng)
            if not (v and nfc(v)!=nfc(s)): continue
            # second pass for multi: apply another word corruptor on the corrupted text
            if rng.random()<multi_frac:
                k=int(rng.random()*len(wt))
                v2=go.apply_word_corruptor(v, go.CORRUPT[wt[k]], rng)
                if v2 and nfc(v2)!=nfc(v): v=v2
            cnt=ve.error_count_label(v, s)            # true single/multi by the labeler
            rows.append({"input":v,"expected":s,"noise_type":ntype,"count":cnt}); made+=1
        counts[ntype]=made
    return rows

def build_split(clean, n_err, leak, seed, name):
    # 50/50 typing/OCR of the error rows
    n_typ=n_err//2; n_ocr=n_err-n_typ
    typ,_,_ = bb.build_typing_pool(clean, n_typ, leak, seed=seed)
    used={nfc(r["expected"]).strip() for r in typ} | {nfc(r["input"]).strip() for r in typ}
    ocr = ocr_pool(clean, n_ocr, leak|used, seed=seed+1)
    err = typ+ocr
    used |= {nfc(r["expected"]).strip() for r in ocr} | {nfc(r["input"]).strip() for r in ocr}
    # identity 18% of total
    n_id=round(0.18/0.82*len(err))
    pool=[s for s in clean if s not in leak and s not in used]
    random.Random(seed+2).shuffle(pool)
    ident=[{"input":s,"expected":s,"noise_type":"identity","count":"identity"} for s in pool[:n_id]]
    rows=err+ident
    random.Random(seed+3).shuffle(rows)
    print(f"  {name}: typing {len(typ)} + OCR {len(ocr)} + identity {len(ident)} = {len(rows)}")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-err", type=int, default=9300)
    ap.add_argument("--val-err", type=int, default=1800)
    ap.add_argument("--test-err", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    a=ap.parse_args()

    clean=load_corpus(["baochi/wiki_clean.txt","corpus/sentences_clean.txt"])
    random.Random(a.seed).shuffle(clean)
    n=len(clean)
    # disjoint partitions
    test_c = clean[:n//6]
    val_c  = clean[n//6:n//3]
    train_c= clean[n//3:]
    print(f"corpus {n} -> train {len(train_c)} / val {len(val_c)} / test {len(test_c)} (disjoint)")

    # build test+val first (smaller), collect their sentences as leak for train
    test_rows = build_split(test_c, a.test_err, set(), a.seed, "TEST")
    test_leak = {nfc(r[k]).strip() for r in test_rows for k in ("input","expected")}
    val_rows  = build_split(val_c, a.val_err, test_leak, a.seed+10, "VAL")
    val_leak  = {nfc(r[k]).strip() for r in val_rows for k in ("input","expected")}
    train_rows= build_split(train_c, a.train_err, test_leak|val_leak, a.seed+20, "TRAIN")

    out={"train":train_rows,"val":val_rows,"test":test_rows}
    for split,rows in out.items():
        p=f"baochi/bal_{split}.jsonl"
        open(p,"w",encoding="utf-8").write("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
        print(f"wrote {p} ({len(rows)})")

    # leak check
    def S(rows): return {nfc(r[k]).strip() for r in rows for k in ("input","expected")}
    st,sv,se=S(train_rows),S(val_rows),S(test_rows)
    print(f"LEAK train<->test {len(st&se)}  train<->val {len(st&sv)}  val<->test {len(sv&se)}")

if __name__=="__main__":
    main()
