"""Relabel some universal questions after a policy change, with an open-weight judge (pusula.judge --refresh).

    # 1) tune the judge on the gold set's dev half: answers vs gold, per question
    python -m pusula.judge --inp gold_v2_dev.jsonl --out dev_judged.jsonl --source relabel --refresh --policy \
        --questions aciliyet,insana_devret,eylem_gerekli
    python -m pusula.relabel score --pred dev_judged.jsonl --source relabel
    # 2) relabel the training data and swap the labels in
    python -m pusula.judge --inp curated/train.jsonl --out train_judged.jsonl --source relabel --refresh --policy \
        --questions aciliyet,insana_devret,eylem_gerekli
    python -m pusula.relabel apply --inp curated/train.jsonl --pred train_judged.jsonl --source relabel \
        --questions aciliyet,insana_devret,eylem_gerekli --out curated_v3/train.jsonl

``apply`` gives the questions their current definitions, makes the new answers the ``final`` labels, and drops the
old labels and teacher probabilities of those questions (they answered the old definitions). An example the judge
did not answer loses those questions instead of keeping stale labels.
"""
from __future__ import annotations

import argparse
import json

from .jsonl import read_examples, write_examples
from .llm import console
from .taxonomy import load_taxonomy

FINAL = "final"


def score(pred: str, source: str, ref: str = "gold") -> dict[str, dict]:
    stats: dict[str, list[int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for ex in read_examples(pred):
        got, want = ex.labels.get(source, {}), ex.labels.get(ref, {})
        for q, v in got.items():
            if q in want:
                s = stats.setdefault(q, [0, 0])
                s[0] += v == want[q]
                s[1] += 1
                key = f"{want[q]}->{v}"
                confusion.setdefault(q, {})[key] = confusion.setdefault(q, {}).get(key, 0) + 1
    return {q: {"acc": round(ok / n, 4), "n": n, "confusion": dict(sorted(confusion[q].items()))}
            for q, (ok, n) in stats.items()}


def apply(inp: str, pred: str, source: str, questions: list[str], out: str) -> int:
    universal = load_taxonomy().universal
    answers = {ex.id: ex.labels.get(source, {}) for ex in read_examples(pred)}
    rows, missing = [], 0
    for ex in read_examples(inp):
        new = answers.get(ex.id, {})
        qs, labels, teacher = dict(ex.questions), {s: dict(v) for s, v in ex.labels.items()}, dict(ex.teacher)
        for q in questions:
            if q not in qs:
                continue
            for votes in labels.values():
                votes.pop(q, None)
            teacher.pop(q, None)
            if q in new:
                qs[q] = universal[q]
                labels.setdefault(FINAL, {})[q] = new[q]
            else:
                qs.pop(q)
                missing += 1
        labels = {s: v for s, v in labels.items() if v}
        if any(labels.get(FINAL, {})):
            rows.append(ex.model_copy(update={"questions": qs, "labels": labels, "teacher": teacher}))
    n = write_examples(out, rows)
    console.print(f"[green]{n} örnek → {out} ({missing} soru cevapsız kaldığı için çıkarıldı)[/green]")
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--ref", default="gold")
    s = sub.add_parser("apply")
    s.add_argument("--inp", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--questions", required=True)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "score":
        print(json.dumps(score(a.pred, a.source, a.ref), ensure_ascii=False, indent=1))
    else:
        apply(a.inp, a.pred, a.source, a.questions.split(","), a.out)


if __name__ == "__main__":
    main()
