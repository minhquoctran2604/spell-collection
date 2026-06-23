"""Quick WER / CER on dumped benchmark outputs (sanity look, NOT the headline).

WER = word-level edit distance(output, expected) / words(expected)
CER = char-level edit distance(output, expected) / chars(expected)
Also reports the input->expected baseline (how wrong the INPUT was) so WER is
read against the floor the model started from, not zero.

Usage:
  python scripts/wer_check.py baochi/bench_v6n_full_outputs.jsonl [more.jsonl ...]
"""
import json, sys, io, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def nfc(s): return unicodedata.normalize("NFC", s or "")

def edit(a, b):
    if a == b: return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (a[i-1] != b[j-1]))
        prev = cur
    return prev[lb]

def rate(path):
    we = wd = ce = cd = 0          # output vs expected
    iwe = iwd = ice = icd = 0      # input  vs expected (baseline floor)
    n = 0
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l); n += 1
        exp = nfc(r["expected"]); out = nfc(r["output"]); inp = nfc(r["input"])
        ew = exp.split(); ow = out.split(); iw = inp.split()
        we += edit(ow, ew); wd += len(ew)
        ce += edit(out, exp); cd += len(exp)
        iwe += edit(iw, ew); iwd += len(ew)
        ice += edit(inp, exp); icd += len(exp)
    return dict(n=n,
                wer=we/wd if wd else 0, cer=ce/cd if cd else 0,
                in_wer=iwe/iwd if iwd else 0, in_cer=ice/icd if icd else 0)

def main():
    paths = sys.argv[1:] or ["baochi/bench_v6n_full_outputs.jsonl"]
    print(f"{'file':<40} {'n':>5} {'WERin':>7} {'WERout':>7} {'CERin':>7} {'CERout':>7}")
    for p in paths:
        try: r = rate(p)
        except FileNotFoundError: print(f"{p:<40} MISSING"); continue
        name = p.split('/')[-1].replace('bench_','').replace('_outputs.jsonl','')
        print(f"{name:<40} {r['n']:>5} {r['in_wer']:>7.4f} {r['wer']:>7.4f} "
              f"{r['in_cer']:>7.4f} {r['cer']:>7.4f}")
    print("\nWERin/CERin = input vs gold (error floor). WERout/CERout = model output vs gold.")
    print("lower WERout = closer to gold. compare WERout < WERin => model helped.")

if __name__ == "__main__":
    main()
