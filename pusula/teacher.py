"""Stage 3: label questions with the base Decision 3.0 model (vllm-sr/d3).

d3 returns a probability for every option without generating text. Its argmax is the third consensus
vote, its distributions are kept as soft targets, and its accuracy is the "before" baseline that the
Turkish fine-tune is measured against. The model is loaded in 4-bit through the same code path as
training (pusula.d3model), so base and fine-tuned d3 are compared under identical quantization; this also
fits the 26B model on GPUs with less than ~60 GB free.

    python -m pusula.teacher --d3 ~/d3 --inp runs/full/judged.jsonl --out runs/full/labeled.jsonl --sample 2000

With --sample N only the first N examples (in file order; generation cycles through every intent, so they
cover the taxonomy evenly) are labeled by d3; the rest are passed through unchanged so later stages see the
whole dataset.
"""
from __future__ import annotations

import argparse
import time

from .jsonl import JsonlWriter, read_jsonl
from .llm import console
from .schema import Example, argmax_key

TEACHER_MODEL = "vllm-sr/d3"


def label_chunk(decider, examples: list[Example], token_budget: int) -> None:
    """Fills ex.teacher / ex.labels['d3'] for every question of `examples` (original option order)."""
    import torch

    from .d3model import render

    jobs = [(ex, name, q) for ex in examples for name, q in ex.questions.items()]
    ids = decider.encode([render(decider.tokenizer, ex.state, q, decider.codes) for ex, _, q in jobs])
    order = sorted(range(len(jobs)), key=lambda i: len(ids[i]))
    batch: list[int] = []

    def flush():
        if not batch:
            return
        counts = [len(jobs[i][2].option_keys()) for i in batch]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = torch.softmax(decider.logits([ids[i] for i in batch], counts), dim=-1).cpu()
        for row, i in enumerate(batch):
            ex, name, q = jobs[i]
            p = probs[row, : counts[row]].tolist()
            ex.teacher[name] = p
            ex.labels.setdefault("d3", {})[name] = argmax_key(q, p)
        batch.clear()

    width = 0
    for i in order:
        w = max(width, len(ids[i]))
        if batch and w * (len(batch) + 1) > token_budget:
            flush()
            w = len(ids[i])
        batch.append(i)
        width = w
    flush()


def run(args) -> None:
    from .d3model import D3Decider

    existing = list(read_jsonl(args.out))
    done = {row["id"] for row in existing}
    labeled = sum(1 for row in existing if "d3" in row.get("labels", {}))
    todo = [Example(**row) for row in read_jsonl(args.inp) if row["id"] not in done]
    if not todo:
        console.print(f"[green]{args.out}: d3 aşaması tamam ({labeled} örnek d3 etiketli).[/green]")
        return
    budget = max(0, args.sample - labeled) if args.sample else len(todo)
    to_label, passthrough = todo[:budget], todo[budget:]
    t0 = time.time()
    with JsonlWriter(args.out) as writer:
        if to_label:
            console.print(f"[cyan]d3 yükleniyor ({args.d3}, 4-bit)...[/cyan]")
            decider = D3Decider.load(args.d3)
            decider.backbone.eval()
            console.print(f"[cyan]d3: {len(to_label)} örnek etiketlenecek (önceden {labeled})[/cyan]")
            for start in range(0, len(to_label), args.chunk):
                chunk = to_label[start:start + args.chunk]
                label_chunk(decider, chunk, args.token_budget)
                for ex in chunk:
                    writer.write(ex)
                n = start + len(chunk)
                console.print(f"d3: {n}/{len(to_label)} ({n / (time.time() - t0):.2f} örnek/s)")
        for ex in passthrough:
            writer.write(ex)
    console.print(f"[green]✓ {len(to_label)} örnek d3 ile etiketlendi, {len(passthrough)} örnek aynen aktarıldı[/green]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d3", required=True, help="local vllm-sr/d3 snapshot directory")
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=64, help="examples per write")
    p.add_argument("--token-budget", type=int, default=24576, help="padded tokens per forward pass")
    p.add_argument("--sample", type=int, default=0, help="label only the first N examples with d3 (0 = all)")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
