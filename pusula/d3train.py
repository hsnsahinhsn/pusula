"""Turkish fine-tuning of Decision 3.0 (vllm-sr/d3) with QLoRA, and evaluation of base vs fine-tuned.

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


def build_items(examples: list[Example], decider, label_source: str, rng: Optional[random.Random],
                max_prompts: int = 0) -> list[Item]:
    from .d3model import render, shuffled_order

    pairs = [(ex, name) for ex in examples for name in ex.labels.get(label_source, {})]
    if max_prompts and len(pairs) > max_prompts and rng is not None:
        # balance question names: equal share each, leftovers filled at random
        by_name = defaultdict(list)
        for p in pairs:
            by_name[p[1]].append(p)
        share = max_prompts // len(by_name)
        chosen, rest = [], []
        for rows in by_name.values():
            rng.shuffle(rows)
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
        meta.append((name, q.type, shown, len(q.option_keys())))
    ids = decider.encode(prompts)
    return [Item(n, t, i, lab, c) for i, (n, t, lab, c) in zip(ids, meta)]


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


def train(a) -> None:
    import torch

    from .d3model import D3Decider

    out = Path(a.out)
    last = out / "last"
    resume = (last / "trainer_state.json").exists()
    decider = D3Decider.load(a.d3, adapter=str(last) if resume else None, lora_rank=a.lora_rank,
                             gradient_checkpointing=True)
    decider.readout.requires_grad_(True)
    rng = random.Random(a.seed)
    t0 = time.time()
    train_items = build_items(read_examples(Path(a.data) / "train.jsonl"), decider, "final", rng, a.max_prompts)
    val_items = build_items(read_examples(Path(a.data) / "val.jsonl"), decider, "final", None, a.max_val)
    console.print(f"[cyan]{len(train_items)} eğitim / {len(val_items)} doğrulama sorusu "
                  f"({time.time() - t0:.0f}s tokenizasyon)[/cyan]")

    lora = [p for p in decider.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": [decider.readout], "lr": a.readout_lr}],
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
        console.print(f"[yellow]Kaldığı yerden devam: adım {step}/{steps_total}[/yellow]")
    console.print(f"[cyan]{steps_total} adım ({a.accum} mikro-batch/adım), {len(schedule)} mikro-batch[/cyan]")

    decider.backbone.train()
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    running, t_step = 0.0, time.time()
    for mb_index in range(step * a.accum, len(schedule)):
        batch = schedule[mb_index]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = decider.logits([it.ids for it in batch], [it.count for it in batch])
        labels = torch.tensor([it.label for it in batch], device=decider.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=a.label_smoothing) / a.accum
        loss.backward()
        running += loss.item()
        if (mb_index + 1) % a.accum and mb_index + 1 != len(schedule):
            continue
        torch.nn.utils.clip_grad_norm_(lora + [decider.readout], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0:
            rate = a.log_every / (time.time() - t_step)
            console.print(f"adım {step}/{steps_total} loss {running / a.log_every:.4f} ({rate:.2f} adım/s)")
            log.write(json.dumps({"step": step, "loss": running / a.log_every}) + "\n")
            log.flush()
            running, t_step = 0.0, time.time()
        if step % a.save_every == 0 or step == steps_total:
            decider.backbone.eval()
            m = evaluate(decider, val_items, a.token_budget)
            decider.backbone.train()
            console.print(f"[bold]adım {step}[/bold] val macro_acc {m['macro_acc']:.4f} ece {m['ece']:.4f}")
            log.write(json.dumps({"step": step, "val": m}, ensure_ascii=False) + "\n")
            log.flush()
            if m["macro_acc"] > best:
                best = m["macro_acc"]
                decider.save_adapter(out / "best", {"step": step, "val": m})
            decider.save_adapter(last, {"step": step})
            torch.save(opt.state_dict(), last / "optimizer.pt")
            torch.save(sched.state_dict(), last / "scheduler.pt")
            (last / "trainer_state.json").write_text(json.dumps({"step": step, "best": best}))
    (out / "DONE").write_text(json.dumps({"steps": step, "best_val_macro_acc": best}))
    console.print(f"[green]✓ Bitti: en iyi val macro_acc {best:.4f} → {out / 'best'}[/green]")


def run_eval(a) -> None:
    from .d3model import D3Decider

    decider = D3Decider.load(a.d3, adapter=a.adapter)
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
    t.add_argument("--max-val", type=int, default=1500)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lora-rank", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--readout-lr", type=float, default=2e-5)
    t.add_argument("--label-smoothing", type=float, default=0.05)
    t.add_argument("--token-budget", type=int, default=8192, help="padded tokens per micro-batch")
    t.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=17)
    e = sub.add_parser("eval")
    e.add_argument("--d3", required=True)
    e.add_argument("--adapter", default=None)
    e.add_argument("--data", required=True)
    e.add_argument("--label-source", default=None)
    e.add_argument("--token-budget", type=int, default=16384)
    e.add_argument("--out", required=True)
    a = p.parse_args(argv)
    train(a) if a.cmd == "train" else run_eval(a)


if __name__ == "__main__":
    main()
