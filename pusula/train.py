"""Stage 5: distill the consensus/teacher labels into the TabiBERT cross-encoder.

Loss per question = alpha * CE(final label) + (1 - alpha) * KL(d3 distribution || student), the KL
term only where d3 agreed with the final label. Questions with many options are trained on a random
subset (the label plus negatives) so batches stay small; evaluation always uses the full option set.
After training, one temperature per question type is fitted on the validation split.

    python -m pusula.train --data runs/full/final --out runs/full/model
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .jsonl import read_examples
from .llm import console
from .model import BASE_MODEL, PairEncoder, PusulaModel
from .schema import QUESTION_TYPES, Example

NEG = -1e4  # padding logit (finite, so 0 * NEG stays 0 in the KL term)


@dataclass
class Item:
    state: int                 # index into the encoded states
    name: str                  # question name
    qtype: str
    options: list[list[int]]   # token ids per option (question + option text)
    label: int
    soft: Optional[list[float]]


class Encoded:
    """Token ids for a split: each state once, each distinct question's options once."""

    def __init__(self, examples: list[Example], enc: PairEncoder):
        self.states = enc.encode_states([ex.state for ex in examples])
        cache: dict[str, list[list[int]]] = {}
        self.items: list[Item] = []
        for si, ex in enumerate(examples):
            for name, label in ex.labels.get("final", {}).items():
                q = ex.questions[name]
                key = q.model_dump_json()
                if key not in cache:
                    cache[key] = enc.encode_options(q)
                keys = q.option_keys()
                self.items.append(Item(si, name, q.type, cache[key], keys.index(label), ex.teacher.get(name)))


def subsample(item: Item, max_options: int, rng: random.Random) -> tuple[list[int], int, Optional[list[float]]]:
    """Option indices to train on, the label's position among them, and renormalized soft targets."""
    n = len(item.options)
    if n <= max_options:
        idx = list(range(n))
    else:
        others = [i for i in range(n) if i != item.label]
        idx = sorted([item.label, *rng.sample(others, max_options - 1)])
    soft = None
    if item.soft is not None:
        s = [item.soft[i] for i in idx]
        total = sum(s)
        soft = [v / total for v in s] if total > 0 else None
    return idx, idx.index(item.label), soft


def make_batches(data: Encoded, enc: PairEncoder, token_budget: int, max_options: Optional[int],
                 rng: Optional[random.Random]) -> list[list[tuple[Item, list[int], int, Optional[list[float]]]]]:
    rows = []
    for it in data.items:
        if max_options and rng is not None:
            idx, lab, soft = subsample(it, max_options, rng)
        else:
            idx, lab, soft = list(range(len(it.options))), it.label, it.soft
        width = min(len(data.states[it.state]) + max(len(it.options[i]) for i in idx) + 3, enc.max_length)
        rows.append((width, it, idx, lab, soft))
    if rng is not None:
        rng.shuffle(rows)
    batches, chunk = [], 4096
    for c in range(0, len(rows), chunk):
        part = sorted(rows[c:c + chunk], key=lambda r: r[0])
        cur, pairs, width = [], 0, 0
        for w, it, idx, lab, soft in part:
            if cur and (pairs + len(idx)) * max(width, w) > token_budget:
                batches.append(cur)
                cur, pairs, width = [], 0, 0
            cur.append((it, idx, lab, soft))
            pairs += len(idx)
            width = max(width, w)
        if cur:
            batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def forward_batch(model: PusulaModel, enc: PairEncoder, data: Encoded, batch, device):
    seqs, counts = [], []
    for it, idx, _, _ in batch:
        state = data.states[it.state]
        seqs += [enc.join(it.options[i], state) for i in idx]
        counts.append(len(idx))
    ids, mask = enc.pad(seqs, device)
    flat = model(ids, mask)
    k = max(counts)
    logits = torch.full((len(batch), k), NEG, device=device)
    for row, chunk in enumerate(torch.split(flat, counts)):
        logits[row, : len(chunk)] = chunk
    return logits, counts


def batch_loss(logits, batch, alpha: float, device):
    labels = torch.tensor([lab for _, _, lab, _ in batch], device=device)
    ce = F.cross_entropy(logits, labels)
    rows = [r for r, (_, _, _, soft) in enumerate(batch) if soft is not None]
    if not rows or alpha >= 1.0:
        return ce
    target = torch.zeros(len(rows), logits.shape[1], device=device)
    for j, r in enumerate(rows):
        s = batch[r][3]
        target[j, : len(s)] = torch.tensor(s, device=device)
    kl = F.kl_div(F.log_softmax(logits[rows], dim=-1), target, reduction="batchmean")
    return alpha * ce + (1 - alpha) * kl


@torch.no_grad()
def predict(model, enc, data: Encoded, device, token_budget: int):
    """Full-option logits per item, in data.items order: list of (item, logits[n_options])."""
    model.eval()
    out = []
    for batch in make_batches(data, enc, token_budget, None, None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, counts = forward_batch(model, enc, data, batch, device)
        for row, (it, _, _, _) in enumerate(batch):
            out.append((it, logits[row, : counts[row]].float().cpu()))
    return out


def metrics(preds, temperature: Optional[dict] = None, bins: int = 15) -> dict:
    per_q: dict[str, list[int]] = {}
    conf, correct = [], []
    for it, lg in preds:
        t = (temperature or {}).get(it.qtype, 1.0)
        p = torch.softmax(lg / t, dim=-1)
        ok = int(p.argmax().item() == it.label)
        per_q.setdefault(it.name, []).append(ok)
        conf.append(p.max().item())
        correct.append(ok)
    acc = {k: sum(v) / len(v) for k, v in sorted(per_q.items())}
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            ece += len(sel) / len(conf) * abs(sum(correct[i] for i in sel) / len(sel)
                                               - sum(conf[i] for i in sel) / len(sel))
    return {"macro_acc": sum(acc.values()) / max(len(acc), 1), "acc": acc, "ece": ece, "n": len(conf)}


def fit_temperatures(preds) -> dict[str, float]:
    temps = {}
    for qtype in QUESTION_TYPES:
        rows = [(lg, it.label) for it, lg in preds if it.qtype == qtype]
        if len(rows) < 20:
            temps[qtype] = 1.0
            continue
        k = max(len(lg) for lg, _ in rows)
        logits = torch.full((len(rows), k), NEG)
        for i, (lg, _) in enumerate(rows):
            logits[i, : len(lg)] = lg
        labels = torch.tensor([lab for _, lab in rows])
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / log_t.exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        temps[qtype] = float(log_t.exp().clamp(0.05, 20.0).item())
    return temps


def train(args) -> None:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    enc = PairEncoder(tokenizer, max_length=args.max_length)
    t0 = time.time()
    train_data = Encoded(read_examples(Path(args.data) / "train.jsonl"), enc)
    val_data = Encoded(read_examples(Path(args.data) / "val.jsonl"), enc)
    console.print(f"[cyan]Tokenizasyon {time.time() - t0:.0f}s: train {len(train_data.items)} soru, "
                  f"val {len(val_data.items)} soru[/cyan]")

    model = PusulaModel.from_base(args.base_model).to(device)
    model.meta.update({"base_model": args.base_model})
    params = [
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": model.scorer.parameters(), "lr": args.head_lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    steps_per_epoch = len(make_batches(train_data, enc, args.token_budget, args.max_train_options,
                                       random.Random(0)))
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    console.print(f"[cyan]Eğitim: {args.epochs} epoch x ~{steps_per_epoch} adım[/cyan]")

    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    best, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = make_batches(train_data, enc, args.token_budget, args.max_train_options, rng)
        t_epoch, running = time.time(), 0.0
        for i, batch in enumerate(batches, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = forward_batch(model, enc, train_data, batch, device)
            loss = batch_loss(logits, batch, args.alpha, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            if i % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                rate = i / (time.time() - t_epoch)
                console.print(f"epoch {epoch} adım {i}/{len(batches)} loss {avg:.4f} ({rate:.1f} adım/s)")
                log.write(json.dumps({"epoch": epoch, "step": step, "loss": avg}) + "\n")
                log.flush()
        m = metrics(predict(model, enc, val_data, device, args.token_budget))
        console.print(f"[bold]epoch {epoch}[/bold] val macro_acc {m['macro_acc']:.4f} ece {m['ece']:.4f} "
                      f"({time.time() - t_epoch:.0f}s)")
        log.write(json.dumps({"epoch": epoch, "val": m}, ensure_ascii=False) + "\n")
        log.flush()
        if m["macro_acc"] > best:
            best = m["macro_acc"]
            model.save(out / "best", tokenizer, args.max_length)

    model, _, _ = PusulaModel.load(out / "best", device=device)
    preds = predict(model, enc, val_data, device, args.token_budget)
    model.temperature = fit_temperatures(preds)
    m = metrics(preds, model.temperature)
    model.save(out / "best", tokenizer, args.max_length)
    (out / "val_metrics.json").write_text(json.dumps(
        {"temperature": model.temperature, **m}, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓ Model: {out / 'best'} | val macro_acc {m['macro_acc']:.4f} "
                  f"ece {m['ece']:.4f} | T={model.temperature}[/green]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory with train.jsonl and val.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the hard-label CE term")
    p.add_argument("--token-budget", type=int, default=65536, help="padded tokens per batch")
    p.add_argument("--max-train-options", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
