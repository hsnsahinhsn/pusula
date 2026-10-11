"""Stage 4: consensus labels, soft targets and train/val/test splits.

Votes per question come from the generator, the judge and the d3 teacher. A question is kept when at
least two votes agree (for score questions: when all votes lie within one level, taking the median).
The kept label goes to ``labels["final"]``; d3's distribution stays in ``teacher`` as the soft target
when d3 agrees with the final label. Questions without consensus go to ``contested.jsonl``, the most
informative pool for gold labeling.

    python -m pusula.dataset --inp runs/full/labeled.jsonl --out-dir runs/full/final
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .jsonl import read_examples, write_examples
from .llm import console
from .schema import Example, Question

SOURCES = ("generator", "judge", "d3")


def consensus(q: Question, votes: list[str]) -> Optional[str]:
    """Final label from the available votes, or None when they do not agree enough."""
    if not votes:
        return None
    if len(votes) == 1:
        return None  # a single vote is not a consensus
    if q.type == "score":
        levels = sorted(int(v) for v in votes)
        if levels[-1] - levels[0] <= 1:
            return str(levels[len(levels) // 2])
        top, n = Counter(votes).most_common(1)[0]
        return top if n >= 2 else None
    top, n = Counter(votes).most_common(1)[0]
    return top if n >= 2 else None


def apply_consensus(ex: Example, stats: dict) -> tuple[Example, list[str]]:
    final, contested = {}, []
    for name, q in ex.questions.items():
        votes = {s: ex.labels[s][name] for s in SOURCES if name in ex.labels.get(s, {})}
        label = consensus(q, list(votes.values()))
        st = stats[name]
        st["total"] += 1
        for s in votes:
            st[f"has_{s}"] += 1
        for a in SOURCES:
            for b in SOURCES:
                if a < b and a in votes and b in votes:
                    st[f"agree_{a}_{b}"] += votes[a] == votes[b]
        if label is None:
            contested.append(name)
            st["contested"] += 1
            continue
        final[name] = label
        st["kept"] += 1
        if "d3" in votes and votes["d3"] != label:
            ex.teacher.pop(name, None)  # d3 disagrees: train on the hard label only
    ex.labels["final"] = final
    return ex, contested


def split_of(ex_id: str, val: float, test: float) -> str:
    h = int(hashlib.sha1(ex_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def build(inp: str, out_dir: str, val: float = 0.05, test: float = 0.05) -> dict:
    stats: dict = defaultdict(Counter)
    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    contested_rows = []
    for ex in read_examples(inp):
        ex, contested = apply_consensus(ex, stats)
        if contested:
            contested_rows.append(ex.model_copy(update={"meta": {**ex.meta, "contested": contested}}))
        if ex.labels["final"]:
            splits[split_of(ex.id, val, test)].append(ex)
    out = Path(out_dir)
    for name, rows in splits.items():
        write_examples(out / f"{name}.jsonl", rows)
    write_examples(out / "contested.jsonl", contested_rows)
    report = {
        "splits": {k: len(v) for k, v in splits.items()},
        "contested_examples": len(contested_rows),
        "questions": {},
    }
    for name, st in sorted(stats.items()):
        row = {"total": st["total"], "kept": st["kept"], "contested": st["contested"]}
        for a in SOURCES:
            for b in SOURCES:
                if a < b:
                    both = min(st[f"has_{a}"], st[f"has_{b}"])
                    if both:
                        row[f"agree_{a}_{b}"] = round(st[f"agree_{a}_{b}"] / both, 4)
        report["questions"][name] = row
    (out / "consensus.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inp", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val", type=float, default=0.05)
    p.add_argument("--test", type=float, default=0.05)
    a = p.parse_args(argv)
    report = build(a.inp, a.out_dir, a.val, a.test)
    console.print_json(data=report)


if __name__ == "__main__":
    main()
