"""Gold set: export candidates for human labeling and import the labels back.

    python -m pusula.gold export --final runs/full/final --n 400 --out runs/full/gold/candidates.json
    python -m pusula.gold import --candidates runs/full/gold/candidates.json \
        --labels runs/full/gold/labels.json --out runs/full/gold/gold.jsonl

Candidates mix a domain-stratified sample of the test split with contested examples (where the
generator, judge and d3 disagreed). The labeling UI rebuilds routing questions from the taxonomy,
so a human who changes ``alan`` gets the matching ``alt_alan`` / ``niyet`` options.

labels.json: ``[{"id": ..., "labels": {question: option_key}, "skipped": false}, ...]``
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .jsonl import read_examples, write_examples
from .llm import console
from .schema import Example, Question
from .taxonomy import Q_DOMAIN, Q_INTENT, Q_SUBDOMAIN, ROUTING_QUESTIONS, load_taxonomy


def _suggestion(ex: Example) -> dict[str, str]:
    out = {}
    for name in ex.questions:
        for source in ("final", "d3", "judge", "generator"):
            if name in ex.labels.get(source, {}):
                out[name] = ex.labels[source][name]
                break
    return out


def _stratified(examples: list[Example], n: int, rng: random.Random) -> list[Example]:
    by_domain = defaultdict(list)
    for ex in examples:
        by_domain[ex.meta.get("path", "?").split(".")[0]].append(ex)
    for rows in by_domain.values():
        rng.shuffle(rows)
    picked, domains = [], sorted(by_domain)
    while len(picked) < n and any(by_domain.values()):
        for d in domains:
            if by_domain[d] and len(picked) < n:
                picked.append(by_domain[d].pop())
    return picked


def export(final_dir: str, n: int, out: str, contested_share: float = 0.3, seed: int = 11) -> int:
    rng = random.Random(seed)
    final = Path(final_dir)
    test = read_examples(final / "test.jsonl")
    contested = read_examples(final / "contested.jsonl") if (final / "contested.jsonl").exists() else []
    n_cont = min(int(n * contested_share), len(contested))
    rng.shuffle(contested)
    chosen = _stratified(test, n - n_cont, rng) + contested[:n_cont]
    rng.shuffle(chosen)
    rows = []
    for ex in chosen:
        sug = _suggestion(ex)
        rows.append({
            "id": ex.id,
            "state": ex.state,
            "source": "contested" if ex.meta.get("contested") else "test",
            "suggestion": sug,
            "extra_questions": {
                name: q.model_dump(exclude_none=True)
                for name, q in ex.questions.items()
                if name not in ROUTING_QUESTIONS and name not in load_taxonomy().universal
            },
            "example": ex.model_dump(exclude_none=True),
        })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]{len(rows)} aday → {out} ({n_cont} tartışmalı)[/green]")
    return len(rows)


def import_labels(candidates: str, labels: str, out: str) -> int:
    tax = load_taxonomy()
    cands = {c["id"]: c for c in json.loads(Path(candidates).read_text(encoding="utf-8"))}
    gold = []
    for row in json.loads(Path(labels).read_text(encoding="utf-8")):
        if row.get("skipped") or row["id"] not in cands:
            continue
        human = row["labels"]
        ex = Example(**cands[row["id"]]["example"])
        try:
            path = tax.path(human[Q_DOMAIN], human[Q_SUBDOMAIN], human[Q_INTENT])
        except KeyError:
            console.print(f"[yellow]{row['id']}: geçersiz yol, atlandı[/yellow]")
            continue
        questions: dict[str, Question] = tax.questions_for(path)
        questions.update({k: Question(**q) for k, q in cands[row["id"]]["extra_questions"].items()})
        # Other sources stay comparable only on questions that are identical to the gold ones.
        labels_out = {"human": {k: v for k, v in human.items() if k in questions}}
        for source, votes in ex.labels.items():
            kept = {k: v for k, v in votes.items() if k in questions and ex.questions.get(k) == questions[k]}
            if kept and source != "human":
                labels_out[source] = kept
        teacher = {k: p for k, p in ex.teacher.items() if k in questions and ex.questions.get(k) == questions[k]}
        gold.append(Example(id=ex.id, state=ex.state, questions=questions,
                            meta={**ex.meta, "gold": True}, labels=labels_out, teacher=teacher))
    n = write_examples(out, gold)
    console.print(f"[green]{n} altın örnek → {out}[/green]")
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--final", required=True)
    e.add_argument("--n", type=int, default=400)
    e.add_argument("--out", required=True)
    e.add_argument("--contested-share", type=float, default=0.3)
    i = sub.add_parser("import")
    i.add_argument("--candidates", required=True)
    i.add_argument("--labels", required=True)
    i.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "export":
        export(a.final, a.n, a.out, a.contested_share)
    else:
        import_labels(a.candidates, a.labels, a.out)


if __name__ == "__main__":
    main()
