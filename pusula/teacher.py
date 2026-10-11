"""Stage 3: label every question with the teacher model vllm-sr/d3 (Decision 3.0).

d3 returns a probability for every option without generating text; those distributions become the
student's soft targets, and d3's argmax is the third consensus vote.

    python -m pusula.teacher --inp runs/full/judged.jsonl --out runs/full/labeled.jsonl
"""
from __future__ import annotations

import argparse
import json
import time

from .jsonl import JsonlWriter, done_ids, read_jsonl
from .llm import console
from .schema import Example, argmax_key, probs_from_d3_answer

TEACHER_MODEL = "vllm-sr/d3"


def load_teacher(model_id: str = TEACHER_MODEL, device: str = "cuda:0", batch_size: int = 16):
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, trust_remote_code=True, device=device, batch_size=batch_size)


def label_example(model, ex: Example) -> tuple[Example, dict]:
    result = model.system_one(
        state=ex.state, questions={name: q.to_d3() for name, q in ex.questions.items()}
    )
    answers = result["answers"]
    votes = {}
    for name, q in ex.questions.items():
        ans = answers.get(name)
        if not ans:
            continue
        try:
            probs = probs_from_d3_answer(q, ans)
        except (KeyError, ValueError) as exc:
            console.print(f"[yellow]{ex.id}.{name}: d3 cevabı kullanılamadı ({exc})[/yellow]")
            continue
        ex.teacher[name] = probs
        votes[name] = argmax_key(q, probs)
    ex.labels["d3"] = votes
    return ex, result


def run(args) -> None:
    done = done_ids(args.out)
    todo = [Example(**row) for row in read_jsonl(args.inp) if row["id"] not in done]
    if args.limit:
        todo = todo[: max(0, args.limit - len(done))]
    if not todo:
        console.print(f"[green]{args.out}: öğretmen etiketlemesi tamam.[/green]")
        return
    console.print(f"[cyan]d3 yükleniyor ({args.model})...[/cyan]")
    model = load_teacher(args.model, args.device, args.batch_size)
    console.print(f"[cyan]Öğretmen: {len(todo)} örnek (önceden biten {len(done)})[/cyan]")
    t0 = time.time()
    with JsonlWriter(args.out) as writer:
        for i, ex in enumerate(todo, 1):
            try:
                ex, raw = label_example(model, ex)
            except Exception as exc:  # e.g. a state over d3's input limit
                console.print(f"[red]{ex.id}: d3 hatası ({exc})[/red]")
                continue
            if i == 1 and args.raw_log:
                with open(args.raw_log, "w", encoding="utf-8") as f:
                    json.dump(raw, f, ensure_ascii=False, indent=2, default=str)
            writer.write(ex)
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                console.print(f"öğretmen: {i}/{len(todo)} ({rate:.2f} örnek/s)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16, help="d3 questions per forward pass")
    p.add_argument("--limit", type=int, default=0, help="stop after this many labeled examples (0 = all)")
    p.add_argument("--raw-log", default=None, help="write the first raw d3 response here")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
