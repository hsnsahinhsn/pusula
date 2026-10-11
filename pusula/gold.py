"""Gold set: blind candidates for an independent annotator, and importing their labels.

    python -m pusula.gold export --final runs/full/final --n 400 --out runs/full/gold
    # -> gold/blind.json (state + extra questions only, no labels) and gold/candidates.jsonl
    python -m pusula.gold import --candidates runs/full/gold/candidates.jsonl \
        --labels runs/full/gold/labels.json --out runs/full/gold/gold.jsonl

Candidates mix a domain-stratified sample of the test split with contested examples (where the
generator, judge and d3 disagreed). The annotator (a strong model from a different family than the
generator and teacher, or a person) sees only the blind file and answers the taxonomy's routing
questions, the universal questions and each example's extra questions.

labels.json: ``[{"id": ..., "labels": {question: option_key}, "drop": false, "confidence": "high"}, ...]``
Entries with ``drop`` or ``skipped`` set are left out.
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

GOLD = "gold"


def extra_questions(ex: Example) -> dict[str, Question]:
    """Questions that are neither taxonomy routing nor universal (e.g. ad-hoc ones)."""
    universal = load_taxonomy().universal
    return {k: q for k, q in ex.questions.items() if k not in ROUTING_QUESTIONS and k not in universal}


def blind_rows(examples: list[Example]) -> list[dict]:
    return [{
        "id": ex.id,
        "state": ex.state,
        "extra_questions": {k: q.model_dump(exclude_none=True) for k, q in extra_questions(ex).items()},
    } for ex in examples]


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


def export(final_dir: str, n: int, out_dir: str, contested_share: float = 0.3, seed: int = 11) -> int:
    rng = random.Random(seed)
    final = Path(final_dir)
    test = read_examples(final / "test.jsonl")
    contested = read_examples(final / "contested.jsonl") if (final / "contested.jsonl").exists() else []
    test_ids = {ex.id for ex in test}
    contested = [ex for ex in contested if ex.id not in test_ids]
    n_cont = min(int(n * contested_share), len(contested))
    rng.shuffle(contested)
    chosen = _stratified(test, n - n_cont, rng) + contested[:n_cont]
    rng.shuffle(chosen)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_examples(out / "candidates.jsonl", chosen)
    (out / "blind.json").write_text(json.dumps(blind_rows(chosen), ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"[green]{len(chosen)} aday → {out} ({n_cont} tartışmalı)[/green]")
    return len(chosen)


def import_labels(candidates: str, labels: str, out: str) -> int:
    tax = load_taxonomy()
    cands = {ex.id: ex for ex in read_examples(candidates)}
    gold = []
    for row in json.loads(Path(labels).read_text(encoding="utf-8")):
        if row.get("drop") or row.get("skipped") or row["id"] not in cands:
            continue
        given = row["labels"]
        ex = cands[row["id"]]
        try:
            path = tax.path(given[Q_DOMAIN], given[Q_SUBDOMAIN], given[Q_INTENT])
        except KeyError:
            console.print(f"[yellow]{row['id']}: geçersiz taksonomi yolu, atlandı[/yellow]")
            continue
        questions: dict[str, Question] = tax.questions_for(path)
        questions.update(extra_questions(ex))
        gold_labels = {k: v for k, v in given.items() if k in questions and v in questions[k].option_keys()}
        # Other sources stay comparable only on questions identical to the gold ones
        # (alt_alan / niyet questions change when the gold domain differs).
        labels_out = {GOLD: gold_labels}
        for source, votes in ex.labels.items():
            kept = {k: v for k, v in votes.items() if k in questions and ex.questions.get(k) == questions[k]}
            if kept and source != GOLD:
                labels_out[source] = kept
        teacher = {k: p for k, p in ex.teacher.items() if k in questions and ex.questions.get(k) == questions[k]}
        meta = {**ex.meta, "gold_confidence": row.get("confidence"), "gold_note": row.get("note")}
        gold.append(Example(id=ex.id, state=ex.state, questions=questions, meta=meta,
                            labels=labels_out, teacher=teacher))
    n = write_examples(out, gold)
    console.print(f"[green]{n} altın örnek → {out}[/green]")
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--final", required=True)
    e.add_argument("--n", type=int, default=400)
    e.add_argument("--out", required=True, help="output directory")
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
