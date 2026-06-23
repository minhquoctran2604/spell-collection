"""Slice baochi/bal_test.jsonl into evaluation subsets, all from ONE test file so
slices stay consistent with the headline set.

  bal_test_typing : noise_type starts with 'typ_'  (typing errors)
  bal_test_ocr    : noise_type starts with 'ocr_'  (OCR errors)
  bal_test_single : error_count_label == 'single'  (incl. identity? no -> only single)
  bal_test_multi  : error_count_label == 'multi'

count is RE-MEASURED by vn_errtype (single source of truth), not the stored field,
so the slices are auditable independent of how the set was built.

Usage:  python scripts/slice_test.py [baochi/bal_test.jsonl]
"""
import json, sys, io
from pathlib import Path
import importlib.util
sp = importlib.util.spec_from_file_location("ve", str(Path(__file__).parent / "vn_errtype.py"))
ve = importlib.util.module_from_spec(sp); sp.loader.exec_module(ve)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "baochi/bal_test.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    typ, ocr, single, multi = [], [], [], []
    for r in rows:
        nt = r.get("noise_type", "")
        cnt = ve.error_count_label(r["input"], r["expected"])
        if nt.startswith("typ_"): typ.append(r)
        elif nt.startswith("ocr_"): ocr.append(r)
        if cnt == "single": single.append(r)
        elif cnt == "multi": multi.append(r)
    out = {
        "bal_test_typing": typ, "bal_test_ocr": ocr,
        "bal_test_single": single, "bal_test_multi": multi,
    }
    base = Path(src).parent
    for name, rs in out.items():
        p = base / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as w:
            for r in rs: w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p} ({len(rs)})")
    n = len(rows)
    print(f"source {src}: {n} rows | typing {len(typ)} ocr {len(ocr)} | "
          f"single {len(single)} multi {len(multi)} "
          f"identity {n-len(single)-len(multi)}")

if __name__ == "__main__":
    main()
