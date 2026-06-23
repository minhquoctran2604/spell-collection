"""Build train/val/test all balanced to ONE target distribution, verified by
vn_errtype (the correct labeler). Does NOT trust the generator's own label —
re-measures every produced pair with vn_errtype and bins by the TRUE type.

Strategy per split:
  TYPING errors: corrupt confusion-prone + random words, classify with vn_errtype,
                 keep until each type hits its target share.
  OCR errors:    reuse gen_ocr output, re-binned by vn_errtype.
  identity:      clean keep sentences.
  multi:         a controlled fraction of sentences get 2 edits (error_count=multi).
Anti-leak across all splits.

This module focuses on TYPING balancing (the part that was mislabeled). OCR + mixing
are done by the caller; here we expose build_typing_pool().
"""
import argparse, json, re, sys, io, unicodedata, random
from collections import Counter, defaultdict
from pathlib import Path
import importlib.util
_spec = importlib.util.spec_from_file_location("ve", str(Path(__file__).parent / "vn_errtype.py"))
ve = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ve)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
nfc = ve.nfc

TARGET = {  # typing, from vsec measured with vn_errtype (independent of test)
    "tone": 0.50, "insdel": 0.20, "hat": 0.18, "other": 0.045,
    "sub": 0.03, "swap": 0.02, "space": 0.015,
}

# --- corruptors that TEND to produce a given type (verified after, not trusted) ---
TONE6 = {"a":"áàảãạ","ă":"ắằẳẵặ","â":"ấầẩẫậ","e":"éèẻẽẹ","ê":"ếềểễệ","i":"íìỉĩị",
         "o":"óòỏõọ","ô":"ốồổỗộ","ơ":"ớờởỡợ","u":"úùủũụ","ư":"ứừửữự","y":"ýỳỷỹỵ"}
TONE_OF = {}
for base, marks in TONE6.items():
    TONE_OF[base] = base
    for m in marks: TONE_OF[m] = base
HAT_MAP = {"a":"âă","â":"aă","ă":"aâ","o":"ôơ","ô":"oơ","ơ":"oô","u":"ư","ư":"u","e":"ê","ê":"e","d":"đ","đ":"d"}
PHON = {"s":"x","x":"s","l":"n","n":"l","r":"d","d":"gi","tr":"ch","ch":"tr"}
KEY = {"a":"s","s":"a","n":"m","m":"n","t":"r","r":"t","o":"p","i":"u","u":"i"}

def base_vowel(ch):
    import unicodedata as u
    d = u.normalize("NFD", ch.lower())
    return d[0] if d else ch

def c_tone(w, rng):  # change tone mark on a vowel
    idx=[i for i,c in enumerate(w) if base_vowel(c) in TONE6]
    if not idx: return None
    i=rng.choice(idx); bv=base_vowel(w[i]); marks=TONE6[bv]
    cur=w[i]
    opts=[bv]+list(marks)
    nw=rng.choice([o for o in opts if o!=cur.lower()])
    if w[i].isupper(): nw=nw.upper()
    return w[:i]+nw+w[i+1:]

def c_hat(w, rng):
    idx=[i for i,c in enumerate(w) if base_vowel(c) in HAT_MAP]
    if not idx: return None
    i=rng.choice(idx); bv=base_vowel(w[i]); opts=HAT_MAP[bv]
    nb=rng.choice(opts)
    # keep tone? simplistic: drop to base form of new hat
    nw=nb.upper() if w[i].isupper() else nb
    return w[:i]+nw+w[i+1:]

def c_insdel(w, rng):
    if len(w)<3: return None
    if rng.random()<0.5:
        i=rng.randrange(len(w)); return w[:i]+w[i]+w[i:]   # dup
    i=rng.randrange(len(w)); return w[:i]+w[i+1:]          # del

def c_other(w, rng):
    lw=w.lower()
    for k in ("tr","ch"):
        if lw.startswith(k):
            nw=PHON[k]; return (nw.capitalize() if w[0].isupper() else nw)+w[len(k):]
    if lw and lw[0] in PHON and len(PHON[lw[0]])==1:
        nw=PHON[lw[0]]; return (nw.upper() if w[0].isupper() else nw)+w[1:]
    return None

def c_sub(w, rng):
    idx=[i for i,c in enumerate(w.lower()) if c in KEY]
    if not idx: return None
    i=rng.choice(idx); nw=KEY[w[i].lower()]
    if w[i].isupper(): nw=nw.upper()
    return w[:i]+nw+w[i+1:]

def c_swap(w, rng):
    if len(w)<2: return None
    i=rng.randrange(len(w)-1); l=list(w); l[i],l[i+1]=l[i+1],l[i]; return "".join(l)

CORRUPT = {"tone":c_tone,"hat":c_hat,"insdel":c_insdel,"other":c_other,"sub":c_sub,"swap":c_swap}

def build_typing_pool(clean_sents, n_err, leak, seed=42, multi_frac=0.30, tri_frac=0.10):
    """Return list of {input,expected,noise_type=typ_<truetype>} balanced to TARGET,
    verified by vn_errtype. multi_frac of error sentences get a 2nd edit;
    tri_frac (of those) get a 3rd edit. 2nd/3rd edits pick a RANDOM type (not just
    tone/sub) so multi sentences mirror the real error-type mix, not a tone-heavy bias."""
    rng = random.Random(seed)
    sents = [s for s in clean_sents if 20<=len(s)<=200 and s not in leak]
    rng.shuffle(sents)
    quota = {t: round(n_err*TARGET[t]) for t in TARGET}
    have = Counter(); rows=[]; si=0; STRIP=ve.STRIP
    WORD=re.compile(r"[A-Za-zÀ-ỹ]")
    def toks(s): return re.split(r"(\s+)", s)
    while sum(have.values())<n_err and si<len(sents):
        s=sents[si]; si+=1
        need=[t for t in TARGET if have[t]<quota[t] and t!="space"]
        if not need: break
        t=rng.choice(need)
        tk=toks(s); widx=[i for i,x in enumerate(tk) if WORD.search(x)]
        if not widx: continue
        rng.shuffle(widx); done=False
        for i in widx:
            w=tk[i].strip(STRIP)
            if not w: continue
            out=CORRUPT[t](w, rng)
            if not out or out==w: continue
            # VERIFY with vn_errtype
            true_t=ve.classify_edit(out, w)   # input=out(lỗi), expected=w(đúng)
            if true_t!=t: continue            # gen ko ra đúng loại → bỏ
            if have[true_t]>=quota[true_t]: break
            newtk=tk[:]; newtk[i]=tk[i].replace(w,out,1)
            # optional extra edits on OTHER words for multi (random type, re-verified)
            note="single"
            used_j={i}
            if rng.random()<multi_frac:
                n_extra = 2 if rng.random()<tri_frac else 1   # 2nd, sometimes 3rd
                cand=[j for j in widx if j not in used_j and tk[j].strip(STRIP)]
                rng.shuffle(cand)
                for j in cand:
                    if n_extra<=0: break
                    w2=tk[j].strip(STRIP)
                    t2=rng.choice([x for x in CORRUPT])      # any type, not just tone/sub
                    o2=CORRUPT[t2](w2, rng)
                    if not o2 or o2==w2: continue
                    if ve.classify_edit(o2, w2)=="identity": continue
                    newtk[j]=tk[j].replace(w2,o2,1); used_j.add(j); note="multi"; n_extra-=1
            corrupted="".join(newtk)
            if nfc(corrupted)==nfc(s): continue
            rows.append({"input":corrupted,"expected":s,"noise_type":f"typ_{true_t}","count":note})
            have[true_t]+=1; done=True; break
    # space errors
    sp_q=round(n_err*TARGET["space"])
    while have["space"]<sp_q and si<len(sents):
        s=sents[si]; si+=1; spi=[k for k,c in enumerate(s) if c==" "]
        if not spi: continue
        k=rng.choice(spi); corr=s[:k]+s[k+1:]
        rows.append({"input":corr,"expected":s,"noise_type":"typ_space","count":"single"}); have["space"]+=1
    return rows, have, quota

if __name__=="__main__":
    # quick self-check: build 2000 typing, verify distribution with vn_errtype
    import sys
    clean=[]
    for src in ["baochi/wiki_clean.txt","corpus/sentences_clean.txt"]:
        if Path(src).exists():
            for l in open(src,encoding="utf-8"):
                clean.append(nfc(l.strip()))
        if len(clean)>40000: break
    rows,have,quota=build_typing_pool(clean, 2000, set(), seed=1)
    print(f"built {len(rows)} typing (target 2000)")
    # re-measure with vn_errtype independently
    from difflib import SequenceMatcher
    c=Counter()
    for r in rows:
        a,b=ve.words(r["input"]),ve.words(r["expected"])
        if r["input"].replace(" ","")==r["expected"].replace(" ",""): c["space"]+=1; continue
        for tag,i1,i2,j1,j2 in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            if tag=="replace":
                for x,y in zip(a[i1:i2],b[j1:j2]): c[ve.classify_edit(x,y)]+=1
    n=sum(c.values())
    print("VERIFIED distribution (vn_errtype):")
    for t in TARGET:
        print(f"  {t:<8}{c.get(t,0)/n*100:5.1f}%  (target {TARGET[t]*100:.0f}%)")
