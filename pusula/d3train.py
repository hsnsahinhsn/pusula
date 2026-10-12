"""Turkish fine-tuning of Decision 3.0 (vllm-sr/d3) with LoRA (BF16) or QLoRA (4-bit), and evaluation.

    # train LoRA adapters + the code readout on the consensus Turkish data
    python -m pusula.d3train train --d3 ~/hf/d3 --data runs/full/final --out runs/full/pusula-d3

    # evaluate (base d3 when --adapter is omitted) on any labeled split, e.g. the gold benchmark
    python -m pusula.d3train eval --d3 ~/hf/d3 --adapter runs/full/pusula-d3/best \
        --data runs/full/gold/gold.jsonl --out runs/full/reports/pusula_d3_gold.json

Training is resumable: `<out>/last` holds the adapter, readout, optimizer and step, and a restart skips
the micro-batches already done (batch order is fixed by the seed). Choice options are shown in a random
order per sample so the model learns the options, not their positions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .jsonl import read_examples
from .llm import console
from .schema import Example


@dataclass
class Item:
    name: str
    qtype: str
    ids: list[int]
    label: int          # index of the label among the options as shown
    count: int          # number of options
    soft: Optional[list[float]] = None  # teacher distribution in the shown order (knowledge distillation)


def build_items(examples: list[Example], decider, label_source: str, rng: Optional[random.Random],
                max_prompts: int = 0, weights: Optional[dict[str, float]] = None) -> list[Item]:
    from .d3model import render, shuffled_order

    pairs = [(ex, name) for ex in examples for name in ex.labels.get(label_source, {})]
    if max_prompts and len(pairs) > max_prompts and rng is not None:
        # balance question names: a share proportional to each name's weight (default 1), leftovers at random
        by_name = defaultdict(list)
        for p in pairs:
            by_name[p[1]].append(p)
        w = {name: (weights or {}).get(name, 1.0) for name in by_name}
        total = sum(w.values())
        chosen, rest = [], []
        for name in sorted(by_name):
            rows = by_name[name]
            rng.shuffle(rows)
            share = int(max_prompts * w[name] / total)
            chosen += rows[:share]
            rest += rows[share:]
        rng.shuffle(rest)
        pairs = chosen + rest[: max_prompts - len(chosen)]
    prompts, meta = [], []
    for ex, name in pairs:
        q = ex.questions[name]
        order = shuffled_order(q, rng) if rng is not None else None
        label = q.option_keys().index(ex.labels[label_source][name])
        shown = order.index(label) if order is not None else label
        prompts.append(render(decider.tokenizer, ex.state, q, decider.codes, order))
        soft = ex.teacher.get(name)
        if soft is not None and order is not None:
            soft = [soft[i] for i in order]
        meta.append((name, q.type, shown, len(q.option_keys()), soft))
    ids = decider.encode(prompts)
    return [Item(n, t, i, lab, c, s) for i, (n, t, lab, c, s) in zip(ids, meta)]


def smoothed_ce(logits, labels, counts, eps: float):
    """Cross-entropy with label smoothing spread over each question's OWN options only.

    torch's built-in label_smoothing also puts mass on the padded (-1e4) codes of shorter questions, which
    explodes the loss and pushes every real code down."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    valid = torch.arange(logits.shape[1], device=logits.device)[None] < torch.as_tensor(counts, device=logits.device)[:, None]
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    smooth = -(logp.masked_fill(~valid, 0.0).sum(1) / valid.sum(1))
    return ((1 - eps) * nll + eps * smooth).mean()


def distill_kl(logits, batch):
    """KL(teacher || student) over each question's own options, for the items that carry a teacher distribution."""
    import torch

    rows = [r for r, it in enumerate(batch) if it.soft is not None]
    if not rows:
        return logits.sum() * 0.0
    k = logits.shape[1]
    target = torch.zeros((len(rows), k), device=logits.device)
    for j, r in enumerate(rows):
        target[j, : batch[r].count] = torch.as_tensor(batch[r].soft, device=logits.device)
    logp = torch.log_softmax(logits[rows].float(), dim=-1)
    valid = target > 0
    return (target[valid] * (target[valid].log() - logp[valid])).sum() / len(rows)


def micro_batches(items: list[Item], token_budget: int, rng: Optional[random.Random]) -> list[list[Item]]:
    rows = list(items)
    if rng is not None:
        rng.shuffle(rows)
    batches = []
    for c in range(0, len(rows), 2048):
        part = sorted(rows[c:c + 2048], key=lambda it: len(it.ids))
        cur, width = [], 0
        for it in part:
            w = max(width, len(it.ids))
            if cur and w * (len(cur) + 1) > token_budget:
                batches.append(cur)
                cur, w = [], len(it.ids)
            cur.append(it)
            width = w
        if cur:
            batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def evaluate(decider, items: list[Item], token_budget: int) -> dict:
    import torch

    per_q: dict[str, list[int]] = defaultdict(list)
    conf, ok_all = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in micro_batches(items, token_budget, None):
            probs = torch.softmax(decider.logits([it.ids for it in batch], [it.count for it in batch]), dim=-1)
            for it, p in zip(batch, probs):
                ok = int(p.argmax().item() == it.label)
                per_q[it.name].append(ok)
                conf.append(p.max().item())
                ok_all.append(ok)
    acc = {k: round(sum(v) / len(v), 4) for k, v in sorted(per_q.items())}
    bins, ece = 15, 0.0
    for b in range(bins):
        sel = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins]
        if sel:
            ece += len(sel) / len(conf) * abs(sum(ok_all[i] for i in sel) / len(sel) - sum(conf[i] for i in sel) / len(sel))
    return {"n": len(ok_all), "macro_acc": round(sum(acc.values()) / max(len(acc), 1), 4),
            "micro_acc": round(sum(ok_all) / max(len(ok_all), 1), 4), "ece": round(ece, 4), "acc": acc}


def _precision(a) -> Optional[bool]:
    return {"4bit": True, "bf16": False}.get(a.precision)  # "auto" -> None (decided by GPU memory)


def _dist():
    """(rank, world, device) — torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; plain python runs single-GPU."""
    import os

    import torch
    import torch.distributed as dist

    world = int(os.getenv("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    local = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    return (dist.get_rank() if world > 1 else 0), world, f"cuda:{local}"


def train(a) -> None:
    """Single- or multi-GPU (torchrun --nproc-per-node N): every GPU holds a full d3 copy, each optimizer
    step's micro-batches are split across GPUs and the LoRA/readout gradients are averaged."""
    import torch
    import torch.distributed as dist

    from .d3model import D3Decider

    rank, world, device = _dist()
    main = rank == 0
    say = console.print if main else (lambda *x, **k: None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last"
    resume = (last / "trainer_state.json").exists()
    if a.full:  # whole model (Pusula Mini); a resume reloads the saved full model
        decider = D3Decider.load(str(last) if resume else a.d3, full_finetune=True, load_in_4bit=False,
                                 gradient_checkpointing=not a.no_grad_ckpt, device=device)
    else:
        decider = D3Decider.load(a.d3, adapter=str(last) if resume else None, lora_rank=a.lora_rank,
                                 gradient_checkpointing=not a.no_grad_ckpt, load_in_4bit=_precision(a), device=device)
    decider.readout.requires_grad_(True)
    rng = random.Random(a.seed)
    t0 = time.time()
    weights = {name: a.focus_weight for name in a.focus.split(",") if name} if a.focus else None
    train_items = build_items(read_examples(Path(a.data) / "train.jsonl"), decider, "final", rng, a.max_prompts,
                              weights)
    val_items = build_items(read_examples(Path(a.data) / "val.jsonl"), decider, "final", None)
    if a.max_val and len(val_items) > a.max_val:  # fixed subset so every evaluation is comparable
        val_items = random.Random(0).sample(val_items, a.max_val)
    say(f"[cyan]{len(train_items)} eğitim / {len(val_items)} doğrulama sorusu "
        f"({time.time() - t0:.0f}s tokenizasyon), {world} GPU[/cyan]")

    params = [p for p in decider.backbone.parameters() if p.requires_grad] + [decider.readout]
    opt = torch.optim.AdamW([{"params": params[:-1], "lr": a.lr}, {"params": [decider.readout], "lr": a.readout_lr}],
                            weight_decay=0.0)
    epochs_batches = [micro_batches(train_items, a.token_budget, random.Random(a.seed + e)) for e in range(a.epochs)]
    schedule = [b for eb in epochs_batches for b in eb]
    steps_total = math.ceil(len(schedule) / a.accum)
    warmup = max(1, int(0.03 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * max(
        0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total)))))
    step, best = 0, -1.0
    if resume:
        state = json.loads((last / "trainer_state.json").read_text())
        step, best = state["step"], state.get("best", -1.0)
        opt.load_state_dict(torch.load(last / "optimizer.pt", map_location="cpu"))
        sched.load_state_dict(torch.load(last / "scheduler.pt", map_location="cpu"))
        say(f"[yellow]Kaldığı yerden devam: adım {step}/{steps_total}[/yellow]")
    say(f"[cyan]{steps_total} adım ({a.accum} mikro-batch/adım), {len(schedule)} mikro-batch[/cyan]")

    decider.backbone.train()
    log = (out / "train_log.jsonl").open("a", encoding="utf-8") if main else None
    running, t_step = 0.0, time.time()
    while step < steps_total:
        step_batches = schedule[step * a.accum:(step + 1) * a.accum]
        step_loss = 0.0
        for batch in step_batches[rank::world]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = decider.logits([it.ids for it in batch], [it.count for it in batch])
            labels = torch.tensor([it.label for it in batch], device=decider.device)
            loss = smoothed_ce(logits, labels, [it.count for it in batch], a.label_smoothing)
            if a.distill_alpha < 1.0:
                loss = a.distill_alpha * loss + (1 - a.distill_alpha) * distill_kl(logits, batch)
            (loss / len(step_batches)).backward()
            step_loss += loss.item() / len(step_batches)
        if world > 1:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)  # sum of per-GPU partial means = mean over the step's micro-batches
            t = torch.tensor([step_loss], device=decider.device)
            dist.all_reduce(t)  # only this step's partial losses: the running sum must not be reduced again
            step_loss = t.item()
        running += step_loss
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0:
            rate = a.log_every / (time.time() - t_step)
            say(f"adım {step}/{steps_total} loss {running / a.log_every:.4f} ({rate:.2f} adım/s)")
            if main:
                log.write(json.dumps({"step": step, "loss": running / a.log_every}) + "\n")
                log.flush()
            running, t_step = 0.0, time.time()
        if step % a.save_every == 0 or step == steps_total:
            if main:
                decider.backbone.eval()
                m = evaluate(decider, val_items, a.token_budget)
                decider.backbone.train()
                say(f"[bold]adım {step}[/bold] val macro_acc {m['macro_acc']:.4f} ece {m['ece']:.4f}")
                log.write(json.dumps({"step": step, "val": m}, ensure_ascii=False) + "\n")
                log.flush()
                if m["macro_acc"] > best:
                    best = m["macro_acc"]
                    decider.save_adapter(out / "best", {"step": step, "val": m})
                decider.save_adapter(last, {"step": step})
                torch.save(opt.state_dict(), last / "optimizer.pt")
                torch.save(sched.state_dict(), last / "scheduler.pt")
                (last / "trainer_state.json").write_text(json.dumps({"step": step, "best": best}))
            if world > 1:
                dist.barrier()
    if main:
        (out / "DONE").write_text(json.dumps({"steps": step, "best_val_macro_acc": best}))
        say(f"[green]✓ Bitti: en iyi val macro_acc {best:.4f} → {out / 'best'}[/green]")
    if world > 1:
        dist.destroy_process_group()


def run_eval(a) -> None:
    from .d3model import D3Decider

    decider = D3Decider.load(a.d3, adapter=a.adapter, load_in_4bit=_precision(a))
    decider.backbone.eval()
    examples = read_examples(a.data)
    source = a.label_source or ("gold" if any("gold" in ex.labels for ex in examples) else "final")
    items = build_items(examples, decider, source, None)
    report = {"model": "pusula-d3" if a.adapter else "d3-base", "adapter": a.adapter, "data": a.data,
              "reference": source, **evaluate(decider, items, a.token_budget)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print_json(data=report)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--d3", required=True, help="local vllm-sr/d3 snapshot directory")
    t.add_argument("--data", required=True, help="directory with train.jsonl / val.jsonl")
    t.add_argument("--out", required=True)
    t.add_argument("--max-prompts", type=int, default=40000, help="training questions (0 = all)")
    t.add_argument("--focus", default="", help="comma-separated question names that get a larger share "
                                                 "of --max-prompts (e.g. where base d3 is weakest)")
    t.add_argument("--focus-weight", type=float, default=2.0)
    t.add_argument("--max-val", type=int, default=1500)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lora-rank", type=int, default=32)
    t.add_argument("--full", action="store_true", help="train the whole model instead of LoRA (small d3-format "
                                                        "models such as Pusula Mini; --d3 is the model directory)")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--readout-lr", type=float, default=2e-5)
    t.add_argument("--label-smoothing", type=float, default=0.05)
    t.add_argument("--token-budget", type=int, default=8192, help="padded tokens per micro-batch")
    t.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    t.add_argument("--no-grad-ckpt", action="store_true",
                   help="keep activations instead of recomputing them (faster, needs more GPU memory; pair with a "
                        "smaller --token-budget and a larger --accum to keep the same tokens per step)")
    t.add_argument("--distill-alpha", type=float, default=1.0,
                   help="weight of the hard-label loss; the rest is KL to the teacher distribution (1 = no distillation)")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--precision", choices=["auto", "4bit", "bf16"], default="auto",
                   help="d3 weights: bf16 (LoRA, needs ~80 GB GPU) or 4bit (QLoRA); auto picks by GPU memory")
    e = sub.add_parser("eval")
    e.add_argument("--d3", required=True)
    e.add_argument("--adapter", default=None)
    e.add_argument("--data", required=True)
    e.add_argument("--label-source", default=None)
    e.add_argument("--token-budget", type=int, default=16384)
    e.add_argument("--out", required=True)
    e.add_argument("--precision", choices=["auto", "4bit", "bf16"], default="auto")
    a = p.parse_args(argv)
    train(a) if a.cmd == "train" else run_eval(a)


if __name__ == "__main__":
    main()
