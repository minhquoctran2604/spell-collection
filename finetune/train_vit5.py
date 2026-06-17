"""Standalone trainer for viT5 (or any T5 with SentencePiece tokenizer).
Independent from train.py so that the BART/BARTpho path stays untouched.

Why a separate module:
  * train.py reweights loss per-token assuming syllable tokenizer alignment
    (BARTpho). Subword (T5) tokenization breaks that 1:1 syllable-to-token
    mapping, so reweighting would be meaningless or wrong.
  * train.py uses processing_class= (transformers >=4.46) and a fast tokenizer.
    viT5's tokenizer needs use_fast=False to avoid a KeyError bug in some
    transformers versions, and we keep the legacy tokenizer= arg for max compat.

This module:
  * AutoTokenizer use_fast=False  -> avoids viT5 tokenizer bug
  * standard Seq2SeqTrainer with tokenizer= kwarg
  * NO custom edit-token reweighting (would require sub-word alignment refactor)
  * SAME compute_metrics shape (corpus + per-noise_type F0.5, identity keep_rate)
    so reports look identical to train.py output and you can compare.

Usage (Kaggle):
  python finetune/train_vit5.py TRAIN.jsonl VAL.jsonl OUT_DIR \
      --model VietAI/vit5-base --epochs 1 --max-train 5000 --eval-steps 100
"""
from __future__ import annotations
import argparse, sys, unicodedata
from collections import defaultdict
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
    T5Tokenizer,
)

# --- shared F0.5 helpers (self-contained copies; ko import từ train.py để giữ độc lập) ---
def _nfc(s: str) -> str: return unicodedata.normalize("NFC", s or "")
def _toks(s: str):
    import re
    return [w for w in re.split(r"\s+", _nfc(s).strip()) if w]
def _edits(a, b):
    """Multiset of word-level edits (a -> b) using SequenceMatcher."""
    from difflib import SequenceMatcher
    A, B = _toks(a), _toks(b)
    ops = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal": continue
        ops.append((tag, tuple(A[i1:i2]), tuple(B[j1:j2])))
    return ops
def prf_counts(src, exp, hyp):
    gold = _edits(src, exp)
    pred = _edits(src, hyp)
    gold_c, pred_c = list(gold), list(pred)
    tp = 0
    for e in list(pred_c):
        if e in gold_c:
            gold_c.remove(e); pred_c.remove(e); tp += 1
    return tp, len(pred_c), len(gold_c)
def fbeta(tp, fp, fn, beta=0.5):
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    b2 = beta*beta
    f = ((1+b2)*p*r/(b2*p + r)) if (b2*p + r) else 0.0
    return p, r, f


def train(train_jsonl, val_jsonl, output_dir, *,
          model_name="VietAI/vit5-base", epochs=1, batch_size=8, grad_accum=8,
          learning_rate=3e-5, max_len=256, eval_subsample=4000,
          max_train=0, eval_steps=200, logging_steps=20, seed=42):
    print(f"Loading JSONL: train={train_jsonl}  val={val_jsonl}", flush=True)
    ds = load_dataset("json", data_files={"train": train_jsonl, "validation": val_jsonl})
    if max_train and len(ds["train"]) > max_train:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(max_train))
        print(f"[SMOKE] train capped to {max_train}", flush=True)

    # Stratified val subsample (mirror train.py behavior)
    if eval_subsample and len(ds["validation"]) > eval_subsample:
        val = ds["validation"].shuffle(seed=seed)
        counts = defaultdict(int)
        for r in val: counts[r.get("noise_type","?")] += 1
        total = sum(counts.values())
        quota = {k: max(1, round(eval_subsample * c / total)) for k, c in counts.items()}
        got = defaultdict(int); keep_idx = []
        for i, r in enumerate(val):
            nt = r.get("noise_type","?")
            if got[nt] < quota.get(nt, 0):
                got[nt] += 1; keep_idx.append(i)
                if len(keep_idx) >= eval_subsample: break
        ds["validation"] = val.select(keep_idx)
        print(f"Val stratified-subsampled to {len(ds['validation'])} per-type={dict(got)}", flush=True)

    print(f"Loading tokenizer & model: {model_name}", flush=True)
    # T5Tokenizer (slow, sentencepiece) directly — bypass AutoTokenizer's
    # convert_to_native_format path that errors with KeyError: 0 on some
    # transformers versions when loading viT5.
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # T5 conventionally uses a task prefix. Spell-correction prompt:
    PREFIX = "sửa lỗi chính tả: "

    val_sources = list(ds["validation"]["input"])
    val_expected = list(ds["validation"]["expected"])
    val_noise = list(ds["validation"]["noise_type"])

    def preprocess(batch):
        inputs = [PREFIX + x for x in batch["input"]]
        model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=batch["expected"], max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        tp = fp = fn = 0
        by_nt = defaultdict(lambda: [0,0,0])
        id_total = id_kept = 0
        n = min(len(decoded), len(val_sources), len(val_expected))
        for i in range(n):
            nt = val_noise[i] if i < len(val_noise) else "unknown"
            if nt == "identity":
                id_total += 1
                if _nfc(decoded[i]) == _nfc(val_sources[i]): id_kept += 1
                _, b, _ = prf_counts(val_sources[i], val_expected[i], decoded[i])
                fp += b; by_nt[nt][1] += b
                continue
            a, b, c = prf_counts(val_sources[i], val_expected[i], decoded[i])
            tp += a; fp += b; fn += c
            by_nt[nt][0] += a; by_nt[nt][1] += b; by_nt[nt][2] += c
        p, r, f = fbeta(tp, fp, fn)
        out = {"f05": f, "precision": p, "recall": r}
        id_keep_rate = id_kept/id_total if id_total else 0.0
        out["identity_keep_rate"] = id_keep_rate
        print("\n  per-noise_type F0.5:", flush=True)
        for nt in sorted(by_nt):
            t, fpp, fnn = by_nt[nt]
            if nt == "identity":
                print(f"    {nt:<18} keep_rate={id_keep_rate:.3f}  ({id_kept}/{id_total})", flush=True)
                continue
            pp, rr, ff = fbeta(t, fpp, fnn)
            print(f"    {nt:<18} P={pp:.3f} R={rr:.3f} F0.5={ff:.3f}  (tp={t} fp={fpp} fn={fnn})", flush=True)
            out[f"f05_{nt}"] = ff
        return out

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps", save_strategy="steps",
        eval_steps=eval_steps, save_steps=eval_steps, logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        warmup_ratio=0.03, weight_decay=0.01, lr_scheduler_type="cosine",
        bf16=True,
        predict_with_generate=True,
        generation_max_length=max_len, generation_num_beams=1,
        group_by_length=False,
        dataloader_num_workers=2,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f05", greater_is_better=True,
        seed=seed, report_to="none",
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,                                # ← legacy arg, tương thích mọi version
        data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print("Training viT5 (no custom reweight)...", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Saved to {output_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_jsonl"); p.add_argument("val_jsonl"); p.add_argument("output_dir")
    p.add_argument("--model", default="VietAI/vit5-base")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--eval-subsample", type=int, default=4000)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(a.train_jsonl, a.val_jsonl, a.output_dir,
          model_name=a.model, epochs=a.epochs, batch_size=a.batch_size,
          grad_accum=a.grad_accum, learning_rate=a.lr, max_len=a.max_len,
          eval_subsample=a.eval_subsample, max_train=a.max_train,
          eval_steps=a.eval_steps, logging_steps=a.logging_steps, seed=a.seed)

if __name__ == "__main__":
    sys.exit(main() or 0)
