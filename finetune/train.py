"""Fine-tune bmd1905/vietnamese-correction-v2 on custom (input, expected) pairs.

Continued full fine-tune of an already-converged Vietnamese spelling corrector,
specialized for OCR errors in traditional-medicine (YHCT) text.

Key design choices (see reports/plans/F05-IMPLEMENTATION-SUMMARY.md + benchmark
breakdown by noise_type):

  * FULL fine-tune (not LoRA): 396M model fits ~6-8GB w/ bf16; the task is to
    shift the model's *when-to-edit* decision boundary (a global behavior), which
    low-rank deltas cannot move. Data volume (896k) is far past the LoRA-helps regime.

  * TOKEN-REWEIGHTED loss: vanilla token CE is dominated by easy copy tokens
    (short sentences, ~1 error each) -> model optimizes copy fluency, stays a
    "lazy corrector" (raw v2 recall 0.443). We up-weight CE on target tokens that
    DIFFER from the source (the actual edits), directly attacking recall.

  * NOISE-TYPE reweight: the per-noise_type breakdown shows v2 is catastrophic on
    no_diacritic (F0.5 0.022) and weak on tone_error (0.331), but strong on
    missing_space (0.750). We scale loss per sample by noise_type so the optimizer
    spends gradient where the model is worst.

  * bf16 (not fp16): wide dynamic range, no overflow. BART is fp16-safe but bf16
    is strictly safer and supported on target GPUs (T4/A100/Blackwell).

  * lr 3e-5 (LOW): model already converged on general VN correction; high lr
    causes catastrophic forgetting of fluency -> precision collapse.

  * epochs 2: corruption is SYNTHETIC (rule-based). Beyond ~2-3 epochs the model
    memorizes the noise functions instead of generalizing (synthetic-GEC gap).

  * F0.5 model selection + early stop on val F0.5 (NOT val loss): val loss keeps
    dropping while real-OCR generalization plateaus. F0.5 is the deployment metric.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


# ---------------------------------------------------------------------------
# Edit-based F0.5 (identical algorithm to scripts/benchmark_common.py so that
# training-time numbers are directly comparable to the benchmark headline).
# ---------------------------------------------------------------------------

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def get_edits(src: str, tgt: str) -> set:
    """Set of non-equal opcodes anchored on SRC coords: (i1, i2, replacement)."""
    src = _nfc(src)
    tgt = _nfc(tgt)
    sm = SequenceMatcher(None, src, tgt, autojunk=False)
    edits = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            edits.add((i1, i2, tgt[j1:j2]))
    return edits


def prf_counts(inp: str, exp: str, act: str) -> tuple[int, int, int]:
    inp_n, exp_n, act_n = _nfc(inp), _nfc(exp), _nfc(act)
    gold = get_edits(inp_n, exp_n)
    if act_n == exp_n:  # short-circuit: perfect output, avoid difflib span asymmetry
        return len(gold), 0, 0
    pred = get_edits(inp_n, act_n)
    return len(gold & pred), len(pred - gold), len(gold - pred)


def fbeta(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * p + r
    f = (1 + b2) * p * r / denom if denom else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Noise-type loss weights. Driven by the raw-v2 per-noise_type F0.5 breakdown:
# worse model behavior -> higher weight (spend gradient where it's weakest).
#   no_diacritic 0.022 -> 3.0   (catastrophic)
#   tone_error   0.331 -> 2.0   (weak)
#   char_subst   0.469 -> 1.3
#   char_swap    0.665 -> 1.0
#   missing_space 0.750 -> 1.0  (already strong)
#   identity     copy   -> 0.6  (raised from 0.3: at 0.3 the model over-corrected
#                                clean sentences (fp=26 on 20 identity val samples,
#                                F0.5=0). identity has NO edits to up-weight at the
#                                token level, so the only lever to teach "leave a
#                                correct sentence alone" is the per-sample weight.
#                                Kept <=1.0 so it does not suppress recall on the
#                                other (edit-bearing) noise types.)
# ---------------------------------------------------------------------------
NOISE_WEIGHTS = {
    "no_diacritic": 3.0,
    "tone_error": 2.0,
    "char_substitution": 1.3,
    "char_swap": 1.0,
    "missing_space": 1.0,
    "identity": 0.6,
    "unknown": 1.0,
}

# Multiplier applied to CE at target token positions that correspond to an EDIT
# (token differs from the aligned source) vs copy positions. This is the core
# recall lever.
EDIT_TOKEN_WEIGHT = 4.0
COPY_TOKEN_WEIGHT = 1.0


class ReweightedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with per-token + per-sample (noise_type) loss reweighting.

    The data collator is configured to keep `token_weight` (per-label-token
    multiplier) and `sample_weight` (per-example multiplier) tensors alongside
    `labels`. compute_loss applies them to a per-token CE.

    Label smoothing is applied INSIDE this loss (the Trainer's built-in
    label_smoother is bypassed by overriding compute_loss).
    """

    label_smoothing: float = 0.1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_weight = inputs.pop("token_weight", None)
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # Per-token CE without reduction, WITH label smoothing folded in so the
        # documented 0.1 smoothing actually takes effect (custom loss bypasses the
        # Trainer's label_smoother). CrossEntropyLoss supports label_smoothing
        # natively (PyTorch >= 1.10).
        loss_fct = nn.CrossEntropyLoss(
            reduction="none", ignore_index=-100, label_smoothing=self.label_smoothing
        )
        vocab = logits.size(-1)
        tok_loss = loss_fct(
            logits.view(-1, vocab), labels.view(-1)
        ).view(labels.size())  # (B, T)

        # Mask of valid (non-pad) label positions.
        valid = (labels != -100).float()  # (B, T)

        if token_weight is not None:
            w = token_weight.to(tok_loss.dtype)
        else:
            w = torch.ones_like(tok_loss)
        if sample_weight is not None:
            w = w * sample_weight.to(tok_loss.dtype).unsqueeze(1)

        w = w * valid
        # Weighted token-mean (normalize by summed weights, not token count, so the
        # effective scale stays stable across batches with different edit density).
        denom = w.sum().clamp_min(1.0)
        loss = (tok_loss * w).sum() / denom

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # During eval, Seq2SeqTrainer calls model.generate(**inputs). Our custom
        # columns (token_weight, sample_weight) survive remove_unused_columns=False
        # and would be forwarded to generate() -> ValueError "model_kwargs not used".
        # They are training-only; strip them before the eval forward/generate.
        inputs = {k: v for k, v in inputs.items() if k not in ("token_weight", "sample_weight")}
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )


class WeightedCollator(DataCollatorForSeq2Seq):
    """DataCollatorForSeq2Seq that also pads `token_weight` and stacks
    `sample_weight`. token_weight is padded with 0.0 (ignored positions)."""

    def __call__(self, features, return_tensors=None):
        token_weights = [f.pop("token_weight") for f in features]
        sample_weights = [f.pop("sample_weight") for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        # Pad token_weight to the same length as labels (right-padded with 0.0).
        label_len = batch["labels"].size(1)
        padded = []
        for tw in token_weights:
            tw = list(tw)[:label_len]
            tw = tw + [0.0] * (label_len - len(tw))
            padded.append(tw)
        batch["token_weight"] = torch.tensor(padded, dtype=torch.float32)
        batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
        return batch


def train(
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 16,
    grad_accum: int = 4,
    learning_rate: float = 3e-5,
    model_name: str = "bmd1905/vietnamese-correction-v2",
    max_len: int = 256,
    eval_subsample: int = 4000,
    max_train: int = 0,
    eval_steps: int = 2000,
    logging_steps: int = 50,
    seed: int = 42,
    label_smoothing: float = 0.1,
    edit_token_weight: float = 4.0,
    identity_weight: float = 0.6,
) -> None:
    if not _IMPORT_OK:
        raise ImportError(
            "HuggingFace libraries not installed. "
            "Run: pip install transformers datasets torch accelerate bitsandbytes"
        )

    print(f"Loading JSONL: train={train_jsonl}  val={val_jsonl}", flush=True)
    ds = load_dataset(
        "json", data_files={"train": train_jsonl, "validation": val_jsonl}
    )

    # Smoke test: cap train set to a small slice to verify the pipeline end-to-end
    # (no crash, weights flow, F0.5 metric fires) before committing to the full run.
    if max_train and len(ds["train"]) > max_train:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(max_train))
        print(f"[SMOKE TEST] train capped to {max_train} examples.", flush=True)

    # Optionally subsample val for in-loop generation (full 49k is slow to generate).
    # Stratify by noise_type so the in-loop F0.5 stays representative of the full
    # distribution (no_diacritic is only ~9% but it's where the model is weakest;
    # a uniform random subsample could under-represent it and hide regressions).
    if eval_subsample and len(ds["validation"]) > eval_subsample:
        val = ds["validation"].shuffle(seed=seed)
        nts = val["noise_type"]
        total = len(val)
        # proportional quota per noise_type
        from collections import Counter
        counts = Counter(nts)
        keep_idx: list[int] = []
        per_type_taken: dict[str, int] = {k: 0 for k in counts}
        quota = {k: max(1, round(eval_subsample * c / total)) for k, c in counts.items()}
        for i, nt in enumerate(nts):
            if per_type_taken[nt] < quota[nt]:
                keep_idx.append(i)
                per_type_taken[nt] += 1
            if len(keep_idx) >= eval_subsample:
                break
        ds["validation"] = val.select(keep_idx)
        print(
            f"Val stratified-subsampled to {len(ds['validation'])} for in-loop F0.5 "
            f"(per-type: {dict(per_type_taken)}).",
            flush=True,
        )

    print(f"Loading tokenizer & model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Keep raw source text for F0.5 (needs input string, not just token ids).
    val_sources = list(ds["validation"]["input"])
    val_expected = list(ds["validation"]["expected"])
    val_noise = list(ds["validation"]["noise_type"])

    # Per-run loss knobs picked up by the `preprocess` closure below. Locals so the
    # module-level constants stay as defaults/fallback while each run can override.
    edit_w = edit_token_weight
    copy_w = COPY_TOKEN_WEIGHT
    noise_weights = dict(NOISE_WEIGHTS)
    noise_weights["identity"] = identity_weight
    print(f"edit_token_weight = {edit_w}  identity_weight = {identity_weight}", flush=True)

    def preprocess(examples):
        inputs = examples["input"]
        targets = examples["expected"]
        model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_len, truncation=True)
        label_ids = labels["input_ids"]
        model_inputs["labels"] = label_ids

        # Per-token weight: align source<->target at CHARACTER level via difflib,
        # then map target chars that fall inside an edit span to the target tokens
        # covering them. Simpler & robust: recompute target-token edit mask by
        # decoding each target token's char span. We approximate at the token level
        # by re-tokenizing and comparing the target token sequence against the
        # source token sequence with SequenceMatcher (token-level alignment).
        src_tok = tokenizer(inputs, max_length=max_len, truncation=True)["input_ids"]
        token_weights = []
        sample_weights = []
        for s_ids, t_ids, nt in zip(src_tok, label_ids, examples["noise_type"]):
            sm = SequenceMatcher(None, s_ids, t_ids, autojunk=False)
            tw = [copy_w] * len(t_ids)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    for j in range(j1, j2):  # target token positions that are edits
                        if 0 <= j < len(tw):
                            tw[j] = edit_w
            token_weights.append(tw)
            sample_weights.append(noise_weights.get(nt, 1.0))
        model_inputs["token_weight"] = token_weights
        model_inputs["sample_weight"] = sample_weights
        return model_inputs

    print("Tokenizing + computing edit/noise weights...", flush=True)
    tokenized = ds.map( preprocess, batched=True, remove_columns=ds["train"].column_names)

    # compute_metrics: corpus-level F0.5 over the (subsampled) val set, PLUS a
    # per-noise_type breakdown. The headline f05 can mask where gains land (e.g.
    # total rises because missing_space got better while no_diacritic stays dead).
    # Tracking per-type P/R/F0.5 every eval shows whether the model is fixing the
    # weak categories (no_diacritic 0.022, tone_error 0.331) or just polishing the
    # already-strong ones. Per-type metrics are logged as f05_<noise_type>.
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Align with stored sources/expected by index (eval set order preserved).
        tp = fp = fn = 0
        # per-noise_type accumulators: nt -> [tp, fp, fn]
        from collections import defaultdict
        by_nt: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        # identity is special: input==expected, so there are NO gold edits -> TP is
        # always 0 -> F0.5 is mathematically always 0 regardless of how well the model
        # behaves. F0.5 is meaningless here. The right metric for identity is the
        # "keep rate": fraction of clean sentences the model left UNCHANGED
        # (actual == input). keep=1.0 is perfect (no over-correction); lower = the
        # model corrupted clean text. Track count + correctly-kept separately.
        id_total = 0
        id_kept = 0
        n = min(len(decoded), len(val_sources), len(val_expected))
        for i in range(n):
            nt = val_noise[i] if i < len(val_noise) else "unknown"
            if nt == "identity":
                id_total += 1
                # kept = model output equals the (already-correct) input, NFC-normalized
                if _nfc(decoded[i]) == _nfc(val_sources[i]):
                    id_kept += 1
                # still count its FP into the corpus totals (over-corrections hurt
                # the overall precision, which is correct), but skip per-type F0.5.
                _, b, _ = prf_counts(val_sources[i], val_expected[i], decoded[i])
                fp += b
                by_nt[nt][1] += b
                continue
            a, b, c = prf_counts(val_sources[i], val_expected[i], decoded[i])
            tp += a
            fp += b
            fn += c
            by_nt[nt][0] += a
            by_nt[nt][1] += b
            by_nt[nt][2] += c
        p, r, f = fbeta(tp, fp, fn)
        out = {"f05": f, "precision": p, "recall": r}
        id_keep_rate = id_kept / id_total if id_total else 0.0
        out["identity_keep_rate"] = id_keep_rate
        # Print a readable per-type table each eval, and surface per-type f05 in the
        # logged metrics (so it lands in trainer_state.json log_history too).
        print("\n  per-noise_type F0.5:", flush=True)
        for nt in sorted(by_nt):
            t, fpp, fnn = by_nt[nt]
            if nt == "identity":
                # report keep_rate instead of the always-zero F0.5
                print(f"    {nt:<18} keep_rate={id_keep_rate:.3f}  "
                      f"({id_kept}/{id_total} kept clean, fp={fpp} over-corrections)", flush=True)
                continue
            pp, rr, ff = fbeta(t, fpp, fnn)
            print(f"    {nt:<18} P={pp:.3f} R={rr:.3f} F0.5={ff:.3f}  (tp={t} fp={fpp} fn={fnn})", flush=True)
            out[f"f05_{nt}"] = ff
        return out

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        # Label smoothing handled inside ReweightedSeq2SeqTrainer.compute_loss
        # (the Trainer's built-in label_smoother is bypassed by the custom loss).
        bf16=True,
        predict_with_generate=True,
        generation_max_length=max_len,
        generation_num_beams=1,
        # group_by_length MUST be False: LengthGroupedSampler reorders the EVAL set
        # too, which would desync decoded preds from val_sources/val_expected (paired
        # by position in compute_metrics) -> F0.5 computed on mismatched triples.
        group_by_length=False,
        # token_weight / sample_weight are NOT in BART.forward signature; with the
        # default remove_unused_columns=True they get stripped before the collator
        # -> reweighting silently dies. Keep them.
        remove_unused_columns=False,
        dataloader_num_workers=2,
        # Keep only 1 checkpoint on disk. Each checkpoint is ~600MB (bf16 weights) +
        # ~1.2GB optimizer state; with limit=2 plus the final save the run exceeded
        # Kaggle's ~20GB /kaggle/working quota and the final save was truncated.
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f05",
        greater_is_better=True,
        seed=seed,
        report_to="none",
    )

    collator = WeightedCollator(tokenizer, model=model)

    trainer = ReweightedSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.label_smoothing = label_smoothing
    print(f"label_smoothing = {label_smoothing}", flush=True)

    print("Starting training (full FT, reweighted loss, F0.5 selection)...", flush=True)
    trainer.train()

    print(f"Saving best model to {output_dir}", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Full FT of seq2seq VN corrector (reweighted, F0.5-selected).")
    p.add_argument("train_jsonl", help="Path to train.jsonl (fields: input, expected, noise_type)")
    p.add_argument("val_jsonl", help="Path to val.jsonl")
    p.add_argument("output_dir", help="Directory to save the trained model")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-subsample", type=int, default=4000, help="Val subset size for in-loop F0.5 (0=full)")
    p.add_argument("--max-train", type=int, default=0, help="Cap train size for smoke test (0=full)")
    p.add_argument("--eval-steps", type=int, default=2000, help="Eval+save every N steps (use ~20 for smoke)")
    p.add_argument("--logging-steps", type=int, default=50, help="Log loss every N steps (use ~5 for smoke)")
    p.add_argument("--model", default="bmd1905/vietnamese-correction-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="CE label smoothing (lower -> less short-sequence/deletion bias; try 0.05 or 0.0)")
    p.add_argument("--edit-token-weight", type=float, default=4.0,
                   help="CE multiplier on edit (changed) target tokens; lower -> less over-editing (try 3.0)")
    p.add_argument("--identity-weight", type=float, default=0.6,
                   help="per-sample loss weight for identity (clean) examples; higher -> less over-correction (try 1.0)")
    args = p.parse_args()

    try:
        train(
            args.train_jsonl,
            args.val_jsonl,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
            model_name=args.model,
            max_len=args.max_len,
            eval_subsample=args.eval_subsample,
            max_train=args.max_train,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            seed=args.seed,
            label_smoothing=args.label_smoothing,
            edit_token_weight=args.edit_token_weight,
            identity_weight=args.identity_weight,
        )
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during training: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
