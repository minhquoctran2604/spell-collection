"""Profile coung21/vi-spelling-correction: does it cover the viwiki test error-words
that our v4 model is missing? If coverage is high, this dataset is gold.

Measures:
  1. WORD COVERAGE: % of viwiki-test error-words present in coung21 error-words
  2. ERROR-TYPE distribution (vs test)
  3. how many NEW error-words coung21 adds beyond our current train_mix4

Usage (Colab/local with `datasets`):
  python scripts/profile_coung21.py --n 30000 --test baochi/viwiki_test_bal.jsonl \
     --train baochi/train_mix4.jsonl
"""
import argparse, json, re, sys, io, unicodedata, collections
from difflib import SequenceMatcher
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
def nfc(s): return unicodedata.normalize("NFC", s or "")
def nfd(s): return unicodedata.normalize("NFD", s)
def sd(s):
    s=nfd(s.lower()); s="".join(c for c in s if unicodedata.category(c)!="Mn"); return s.replace("đ","d")
TONE="̣̀́̃̉"; HAT=set("âăêôơưđÂĂÊÔƠƯĐ")
WS=re.compile(r"\s+"); STR=".,;:!?\"'()[]{}«»…"
def W(s): return [w.strip(STR) for w in WS.split(nfc(s).strip()) if w.strip(STR)]
def ht(s): return any(c in TONE for c in nfd(s))
def fine(a,b):
    if a==b: return None
    if a.replace(" ","")==b.replace(" ",""): return "space"
    wa,wb=W(a),W(b)
    ops=[o for o in SequenceMatcher(None,wa,wb,autojunk=False).get_opcodes() if o[0]!="equal"]
    sg=[o for o in ops if o[0]=="replace" and o[2]-o[1]==1 and o[4]-o[3]==1]
    if not(len(ops)==1 and len(sg)==1): return "multi"
    _,i1,_,j1,_=sg[0]; x,y=wa[i1],wb[j1]; sx,sy=sd(x.lower()),sd(y.lower())
    if sx==sy:
        if ht(y) and not ht(x): return "drop_tone"
        if any(c in HAT for c in x)!=any(c in HAT for c in y): return "hat"
        return "wrong_tone"
    if sorted(sx)==sorted(sy) and len(sx)>1 and sx!=sy: return "swap"
    if abs(len(sx)-len(sy))==1 and (sx in sy or sy in sx): return "insdel"
    if len(sx)==len(sy): return "sub"
    return "other"

def errwords_and_types(pairs):
    """pairs: iterable of (input, expected). return (errword_counter, type_counter)"""
    ew = collections.Counter(); ty = collections.Counter()
    for inp, exp in pairs:
        a, b = W(nfc(inp)), W(nfc(exp))
        for tag,i1,i2,j1,j2 in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            if tag=="replace":
                for x in a[i1:i2]: ew[x.lower()] += 1
        t = fine(nfc(inp), nfc(exp))
        if t: ty[t] += 1
    return ew, ty

def load_jsonl_pairs(path):
    out=[]
    for l in open(path, encoding="utf-8"):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        if r.get("noise_type")=="identity": continue
        out.append((r.get("input",""), r.get("expected","")))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--test", default="baochi/viwiki_test_bal.jsonl")
    ap.add_argument("--train", default="baochi/train_mix4.jsonl")
    ap.add_argument("--repo", default="coung21/vi-spelling-correction")
    a=ap.parse_args()

    test_ew,_ = errwords_and_types(load_jsonl_pairs(a.test))
    train_ew,_ = errwords_and_types(load_jsonl_pairs(a.train))
    print(f"viwiki test error-words: {len(test_ew)}")
    print(f"train_mix4 covers: {len(set(test_ew)&set(train_ew))} = {len(set(test_ew)&set(train_ew))/len(test_ew)*100:.0f}%")

    from datasets import load_dataset
    ds = load_dataset(a.repo, split="train", streaming=True)
    pairs=[]
    # detect field names from first row
    first=next(iter(ds))
    keys=list(first.keys())
    print("coung21 fields:", keys)
    # common: input/output, source/target, noisy/clean
    def pick(r):
        for si,ti in [("input","output"),("source","target"),("noisy","clean"),("text","label")]:
            if si in r and ti in r: return r[si], r[ti]
        # fallback first two str fields
        vals=[v for v in r.values() if isinstance(v,str)]
        return (vals[0], vals[1]) if len(vals)>=2 else ("","")
    pairs.append(pick(first))
    for i,r in enumerate(ds):
        if i>=a.n: break
        pairs.append(pick(r))
    print(f"coung21 sampled: {len(pairs)}")

    co_ew, co_ty = errwords_and_types(pairs)
    n_ty=sum(co_ty.values())
    print("\ncoung21 error-type dist:")
    for t,c in co_ty.most_common(): print(f"  {t:<12}{c/n_ty*100:5.1f}%")

    # coverage analysis
    co_cov = set(test_ew)&set(co_ew)
    print(f"\n### COVERAGE ###")
    print(f"coung21 alone covers test error-words: {len(co_cov)} = {len(co_cov)/len(test_ew)*100:.0f}%")
    union = (set(train_ew)|set(co_ew)) & set(test_ew)
    print(f"train_mix4 + coung21 UNION covers: {len(union)} = {len(union)/len(test_ew)*100:.0f}%")
    # NEW words coung21 brings that train_mix4 lacks (and test needs)
    new_useful = (set(co_ew) & set(test_ew)) - set(train_ew)
    print(f"NEW test-error-words coung21 adds (train_mix4 lacks): {len(new_useful)}")
    print("  examples:", sorted(new_useful, key=lambda w:-test_ew[w])[:15])

if __name__=="__main__":
    main()
