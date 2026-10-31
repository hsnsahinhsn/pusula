"""Turkish fine-tuning of Decision 3.0 (vllm-sr/d3) with LoRA (BF16) or QLoRA (4-bit), and evaluation.

    # train LoRA adapters + the code readout on the consensus Turkish data
    python -m pusula.d3train train --d3 ~/hf/d3 --data runs/full/final --out runs/full/pusula-d3

    # evaluate (base d3 when --adapter is omitted) on any labeled split, e.g. the gold benchmark
    python -m pusula.d3train eval --d3 ~/hf/d3 --adapter runs/full/pusula-d3/best \
        --data runs/full/gold/gold.jsonl --out runs/full/reports/pusula_d3_gold.json

    # CPU gate before a GPU run (tokenizer only): items build, labels are valid, validation is not empty
    python -m pusula.d3train check --d3 ~/hf/d3 --data runs/full/final --eval runs/full/gold/gold.jsonl

Training is resumable: `<out>/last` holds the adapter, readout, optimizer and step, and a restart skips
the micro-batches already done (batch order is fixed by the seed). Choice options are shown in a random
order per sample so the model learns the options, not their positions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
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


BOOST_MAX = 3.0  # class balance: a rare label value gets at most 3 times its natural share (user decision, 21 Oct 2026)


def rare_first(rows: list, label_source: str, rng: random.Random, share: int, boost: float = BOOST_MAX) -> list:
    """One question name's (example, name) pairs, ordered so that the first `share` rows give the rare label values a
    larger share: the values are filled evenly (water-filling), each up to `boost` times its natural share, ties in a
    random order. A share cut from the front keeps more rows of a rare class (e.g. hakaret) without moving the prior far.

    Only for a fixed option set (one question type, the same options in every row: a template). A name whose options
    differ per row, such as the writer's own questions s1-s3 (choice keys almost all unique, mixed with noul and score),
    is shuffled as in v1.0: balancing by answer string would fill its share with one-off choice keys."""
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    kinds = {(p[0].questions[p[1]].type, tuple(p[0].questions[p[1]].option_keys())) for p in rows}
    if share >= n or len(kinds) > 1:
        return rows
    by_label = defaultdict(list)
    for p in rows:  # rows are shuffled: each group is a random sample
        by_label[p[0].labels[label_source][p[1]]].append(p)
    cap = {k: min(len(g), math.ceil(boost * share * len(g) / n)) for k, g in by_label.items()}
    order = list(by_label)
    rng.shuffle(order)
    order.sort(key=lambda k: cap[k])  # stable: equal caps keep the random order
    quota, left = {}, share
    for i, k in enumerate(order):
        quota[k] = min(cap[k], left // (len(order) - i))
        left -= quota[k]
    for k in order:  # what the floors left over, to values below their cap (the caps add up to at least share)
        add = min(cap[k] - quota[k], left)
        quota[k] += add
        left -= add
    chosen = [p for k in order for p in by_label[k][: quota[k]]]
    rng.shuffle(chosen)
    return chosen + [p for k in order for p in by_label[k][quota[k]:]]


def save_due(step: int, steps_total: int, save_every: int, save_minutes: float, since_save_s: float):
    """Checkpoint now? True on the step schedule (the same on every rank), "clock" when only the clock says so (ranks
    must agree on that first), False otherwise."""
    if step % save_every == 0 or step == steps_total:
        return True
    return "clock" if save_minutes and since_save_s >= save_minutes * 60 else False


def _balance(a) -> bool | frozenset:
    """Class balance: off (--no-class-balance), only in --balance-only names (31 Oct 2026: natural sampling for the
    routing and urgency priors, rare values first where the rare value is the point, e.g. eksik_bilgi true), or all."""
    if a.balance_only:
        return frozenset(n for n in a.balance_only.split(",") if n)
    return not a.no_class_balance


def build_items(examples: list[Example], decider, label_source: str, rng: Optional[random.Random],
                max_prompts: int = 0, weights: Optional[dict[str, float]] = None,
                class_balance: bool | frozenset = False) -> list[Item]:
    from .d3model import render, shuffled_order

    pairs = [(ex, name) for ex in examples for name in ex.labels.get(label_source, {})]
    if not pairs:
        return []
    if max_prompts and len(pairs) > max_prompts and rng is not None:
        # balance question names: a share proportional to each name's weight (default 1), leftovers at random;
        # class_balance: inside a fixed-option name the rare label values get a larger share (see rare_first)
        by_name = defaultdict(list)
        for p in pairs:
            by_name[p[1]].append(p)
        w = {name: (weights or {}).get(name, 1.0) for name in by_name}
        total = sum(w.values())
        chosen, rest = [], []
        for name in sorted(by_name):
            rows = by_name[name]
            share = int(max_prompts * w[name] / total)
            if class_balance is True or (class_balance and name in class_balance):
                rows = rare_first(rows, label_source, rng, share)
            else:
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


def val_set(examples: list[Example], decider, seed: int, max_val: int) -> list[Item]:
    """Validation items: options in a fixed-seed shuffled order (a writer that puts its target first must not reward a
    first-option habit), then a fixed subset so every evaluation is comparable."""
    items = build_items(examples, decider, "final", random.Random(seed + 1000))
    if max_val and len(items) > max_val:
        items = random.Random(0).sample(items, max_val)
    return items


class TokenizerDecider:
    """What build_items needs (tokenizer, codes, encode) without the d3 weights: a CPU gate before a GPU run."""

    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer

        from .d3model import load_codes

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.codes = load_codes(model_dir)

    def encode(self, prompts):
        return self.tokenizer(list(prompts), add_special_tokens=False)["input_ids"]


def run_check(a) -> None:
    """CPU gate: the training and evaluation items build exactly as `train` / `eval` build them (labels valid,
    validation not empty). Exits non-zero on any problem; lengths over the token budgets are reported (an item
    longer than the budget runs as a batch of its own)."""
    decider = TokenizerDecider(a.d3)
    report, problems = {}, []

    def lengths(name, items, budget):
        lens = sorted(len(it.ids) for it in items)
        report[name] = {"items": len(items), "max_tokens": lens[-1] if lens else 0,
                        "mean_tokens": round(sum(lens) / max(len(lens), 1), 1),
                        "over_budget": sum(x > budget for x in lens)}

    if a.data:
        weights = {name: a.focus_weight for name in a.focus.split(",") if name} if a.focus else None
        try:
            train_items = build_items(read_examples(Path(a.data) / "train.jsonl"), decider, "final",
                                      random.Random(a.seed), a.max_prompts, weights, _balance(a))
            val_items = val_set(read_examples(Path(a.data) / "val.jsonl"), decider, a.seed, a.max_val)
        except Exception as e:  # noqa: BLE001 - every failure is a gate failure, reported with its cause
            problems.append(f"{a.data}: {type(e).__name__}: {e}")
        else:
            lengths("train", train_items, a.token_budget)
            lengths("val", val_items, a.token_budget)
            if not val_items:
                problems.append(f"{a.data}/val.jsonl: doğrulama sorusu yok")
            if a.focus:
                seen = {it.name for it in train_items}
                missing = [n for n in a.focus.split(",") if n and n not in seen]
                if missing:
                    report["focus_missing"] = missing  # a warning: the share goes to the other names
    for f in a.eval or []:
        try:
            examples = read_examples(f)
            source = "gold" if any("gold" in ex.labels for ex in examples) else "final"
            items = build_items(examples, decider, source, None)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{f}: {type(e).__name__}: {e}")
            continue
        lengths(Path(f).name, items, a.eval_token_budget)
        if not items:
            problems.append(f"{f}: etiketli soru yok")
    report["problems"] = problems
    print(json.dumps(report, ensure_ascii=False))
    if problems:
        raise SystemExit(1)


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


def _swap_in(tmp: Path, dst: Path) -> None:
    """Moves a fully written checkpoint folder into place: a kill at any point leaves ``dst`` or ``dst.old`` complete."""
    old = dst.with_name(dst.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        dst.rename(old)
    tmp.rename(dst)
    shutil.rmtree(old, ignore_errors=True)


def _recover(dst: Path) -> None:
    """Undoes a save that was cut between the two renames of ``_swap_in``; drops a half-written ``.tmp``."""
    old, tmp = dst.with_name(dst.name + ".old"), dst.with_name(dst.name + ".tmp")
    if not dst.exists() and old.exists():
        old.rename(dst)
    elif old.exists():
        shutil.rmtree(old)
    if tmp.exists():
        shutil.rmtree(tmp)


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return path


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
    if main:  # a save cut by a kill: the previous complete checkpoint is used
        _recover(out / "best")
        _recover(last)
    if world > 1:
        dist.barrier()
    resume = (last / "trainer_state.json").exists() and not a.bench
    start = str(last) if resume else a.init_adapter  # continue from a trained Pusula adapter (e.g. v1 -> v2)
    train_ex, val_ex = read_examples(Path(a.data) / "train.jsonl"), read_examples(Path(a.data) / "val.jsonl")
    if not a.bench and not any(ex.labels.get("final") for ex in val_ex):  # before d3 is loaded (GPU time)
        raise SystemExit(f"{Path(a.data) / 'val.jsonl'}: etiketli doğrulama sorusu yok; en iyi kayıt seçilemez")
    decider = D3Decider.load(a.d3, adapter=start, lora_rank=a.lora_rank,
                             gradient_checkpointing=not a.no_grad_ckpt, load_in_4bit=_precision(a), device=device)
    decider.readout.requires_grad_(True)
    rng = random.Random(a.seed)
    t0 = time.time()
    weights = {name: a.focus_weight for name in a.focus.split(",") if name} if a.focus else None
    train_items = build_items(train_ex, decider, "final", rng, a.max_prompts, weights, _balance(a))
    val_items = val_set(val_ex, decider, a.seed, a.max_val)
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
    t_save = t_step
    bench_tokens, t_bench = 0, 0.0
    bad = skipped = 0  # non-finite micro-batch losses / optimizer steps skipped for non-finite gradients
    while step < steps_total:
        step_batches = schedule[step * a.accum:(step + 1) * a.accum]
        if a.bench and step == 1:  # the first step warms up kernels and the allocator
            torch.cuda.synchronize()
            t_bench = time.time()
            torch.cuda.reset_peak_memory_stats()
        if a.bench and step >= 1:
            bench_tokens += sum(len(it.ids) for b in step_batches for it in b)  # real tokens, all GPUs
        step_loss = 0.0
        for batch in step_batches[rank::world]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = decider.logits([it.ids for it in batch], [it.count for it in batch])
            labels = torch.tensor([it.label for it in batch], device=decider.device)
            loss = smoothed_ce(logits, labels, [it.count for it in batch], a.label_smoothing)
            if a.distill_alpha < 1.0:
                loss = a.distill_alpha * loss + (1 - a.distill_alpha) * distill_kl(logits, batch)
            if not torch.isfinite(loss):  # one bad micro-batch must not poison the weights
                bad += 1
                continue
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
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if torch.isfinite(norm):  # same decision on every GPU: gradients were all-reduced
            opt.step()
        else:
            skipped += 1
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if a.bench and step == a.bench + 1:
            torch.cuda.synchronize()
            dt = time.time() - t_bench
            res = {"grad_ckpt": not a.no_grad_ckpt, "token_budget": a.token_budget, "accum": a.accum,
                   "tokens_per_s": round(bench_tokens / dt), "s_per_step": round(dt / a.bench, 2),
                   "est_hours": round(dt / a.bench * steps_total / 3600, 2),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            say(json.dumps(res))
            if main:
                with (out / "bench.jsonl").open("a") as f:
                    f.write(json.dumps(res) + "\n")
            break
        if step % a.log_every == 0:
            rate = a.log_every / (time.time() - t_step)
            say(f"adım {step}/{steps_total} loss {running / a.log_every:.4f} ({rate:.2f} adım/s)"
                + (f" [yellow]atlanan: {bad} mikro-batch, {skipped} adım[/yellow]" if bad or skipped else ""))
            if main:
                log.write(json.dumps({"step": step, "loss": running / a.log_every}) + "\n")
                log.flush()
            running, t_step = 0.0, time.time()
        due = save_due(step, steps_total, a.save_every, 0 if a.bench else a.save_minutes, time.time() - t_save)
        if world > 1 and a.save_minutes and not a.bench and due is not True:
            # the clocks of the ranks may disagree: all ranks reach this line together (the step schedule is shared)
            # and take the same branch (the barrier below)
            flag = torch.tensor([float(due == "clock")], device=decider.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            due = flag.item() > 0
        if not a.bench and due:
            t_save = time.time()
            if main:
                decider.backbone.eval()
                m = evaluate(decider, val_items, a.val_token_budget or a.token_budget)
                decider.backbone.train()
                say(f"[bold]adım {step}[/bold] val macro_acc {m['macro_acc']:.4f} ece {m['ece']:.4f}")
                log.write(json.dumps({"step": step, "val": m}, ensure_ascii=False) + "\n")
                log.flush()
                if m["macro_acc"] > best:  # written to <name>.tmp, then swapped in (a kill never leaves a torn folder)
                    best = m["macro_acc"]
                    tmp = _fresh(out / "best.tmp")
                    decider.save_adapter(tmp, {"step": step, "val": m})
                    _swap_in(tmp, out / "best")
                tmp = _fresh(out / "last.tmp")
                decider.save_adapter(tmp, {"step": step})
                torch.save(opt.state_dict(), tmp / "optimizer.pt")
                torch.save(sched.state_dict(), tmp / "scheduler.pt")
                (tmp / "trainer_state.json").write_text(json.dumps({"step": step, "best": best}))
                _swap_in(tmp, last)
            if world > 1:
                dist.barrier()
    if main and not a.bench:
        (out / "DONE").write_text(json.dumps({"steps": step, "best_val_macro_acc": best}))
        say(f"[green]✓ Bitti: en iyi val macro_acc {best:.4f} → {out / 'best'}[/green]")
    if world > 1:
        dist.destroy_process_group()


def run_eval(a) -> None:
    import torch

    from .d3model import D3Decider

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if len(a.data) != len(a.out):
        raise SystemExit(f"--data ({len(a.data)}) and --out ({len(a.out)}) must pair up")
    # one model load for every set (30 Oct 2026: a load per set cost ~3 min each, ~half of the evaluation time)
    decider = D3Decider.load(a.d3, adapter=a.adapter, load_in_4bit=_precision(a), device=device)
    decider.backbone.eval()
    for data, out_path in zip(a.data, a.out):
        examples = read_examples(data)
        source = a.label_source or ("gold" if any("gold" in ex.labels for ex in examples) else "final")
        items = build_items(examples, decider, source, None)
        report = {"model": "pusula-d3" if a.adapter else "d3-base", "adapter": a.adapter, "data": data,
                  "reference": source, **evaluate(decider, items, a.token_budget)}
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # atomic: a run cut while writing must not leave a partial report that the next run takes as done
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
        console.print(f"EVAL_DONE {out}")
        console.print_json(data=report)


def ddp_probe(dist, device: str, rank: int, world: int, mb: int = 256, small: int = 200) -> tuple[bool, float]:
    """All-reduces one large tensor (3 times) and many small ones (like the per-parameter LoRA gradients), every
    result checked exactly (a broken peer-to-peer path can corrupt data as well as hang). Returns (ok on all ranks,
    GB/s of the large all-reduces)."""
    import torch

    want = world * (world + 1) / 2  # rank r sends r + 1; sums of small integers are exact in float32
    big = torch.full((mb * 2**18,), float(rank + 1), device=device)
    good, t0 = True, time.time()
    for _ in range(3):
        x = big.clone()
        dist.all_reduce(x)
        good &= bool((x == want).all())
    gbps = 3 * big.numel() * 4 / max(time.time() - t0, 1e-9) / 1e9
    for _ in range(small):
        x = torch.full((2**18,), float(rank + 1), device=device)
        dist.all_reduce(x)
        good &= bool((x == want).all())
    flag = torch.tensor([0.0 if good else 1.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() == 0.0, gbps


def run_ddpcheck(a) -> None:
    """Link test before multi-GPU training (under torchrun): rank 0 prints DDP_OK or DDP_BAD, a bad result exits 2.
    A hung link never returns; the caller's timeout catches it."""
    import sys

    import torch.distributed as dist

    rank, world, device = _dist()
    t0 = time.time()
    ok, gbps = ddp_probe(dist, device, rank, world, a.mb, a.small)
    if rank == 0:
        print(f"{'DDP_OK' if ok else 'DDP_BAD'} {world} GPU, {gbps:.1f} GB/s, {time.time() - t0:.1f} s", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    if not ok:
        sys.exit(2)


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
    t.add_argument("--val-token-budget", type=int, default=0,
                   help="padded tokens per validation micro-batch (no gradients, so it can be larger; 0 = --token-budget)")
    t.add_argument("--no-class-balance", action="store_true",
                   help="sample each question name's share at random (v1.0) instead of rare label values first")
    t.add_argument("--balance-only", default="",
                   help="comma-separated names: rare label values first only in these names, the rest at random")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lora-rank", type=int, default=32)
    t.add_argument("--init-adapter", default=None, help="start from this trained adapter (LoRA + readout) instead of base d3")
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
    t.add_argument("--save-minutes", type=float, default=0,
                   help="also validate and save when this many minutes passed since the last save (0 = off)")
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--precision", choices=["auto", "4bit", "bf16"], default="auto",
                   help="d3 weights: bf16 (LoRA, needs ~80 GB GPU) or 4bit (QLoRA); auto picks by GPU memory")
    t.add_argument("--bench", type=int, default=0, help="speed test: time this many optimizer steps after a warm-up "
                                                       "step, append tokens/s and peak memory to OUT/bench.jsonl, "
                                                       "save nothing (see scripts/speedtest.sh)")
    e = sub.add_parser("eval")
    e.add_argument("--d3", required=True)
    e.add_argument("--adapter", default=None)
    e.add_argument("--data", required=True, nargs="+", help="one or more evaluation files (the model loads once)")
    e.add_argument("--label-source", default=None)
    e.add_argument("--token-budget", type=int, default=16384)
    e.add_argument("--out", required=True, nargs="+", help="one report per --data file, same order")
    e.add_argument("--precision", choices=["auto", "4bit", "bf16"], default="auto")
    c = sub.add_parser("check", help="CPU gate before a GPU run: build the items with the d3 tokenizer only")
    c.add_argument("--d3", required=True, help="d3 directory (only the tokenizer and decision_config.json are read)")
    c.add_argument("--data", default=None, help="training directory (train.jsonl / val.jsonl), built as `train` does")
    c.add_argument("--eval", nargs="*", default=None, help="evaluation files, built as `eval` does")
    c.add_argument("--max-prompts", type=int, default=40000)
    c.add_argument("--focus", default="")
    c.add_argument("--focus-weight", type=float, default=2.0)
    c.add_argument("--max-val", type=int, default=1500)
    c.add_argument("--no-class-balance", action="store_true")
    c.add_argument("--balance-only", default="")
    c.add_argument("--seed", type=int, default=17)
    c.add_argument("--token-budget", type=int, default=8192)
    c.add_argument("--eval-token-budget", type=int, default=16384)
    d = sub.add_parser("ddpcheck", help="multi-GPU link test before training (run under torchrun)")
    d.add_argument("--mb", type=int, default=256, help="size of the large all-reduce in MiB")
    d.add_argument("--small", type=int, default=200, help="number of 1 MiB all-reduces")
    a = p.parse_args(argv)
    {"train": train, "eval": run_eval, "check": run_check, "ddpcheck": run_ddpcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
