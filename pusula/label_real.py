"""Hierarchical judge labeling for real messages (domain, then subdomain, then intent).

The open-weight judge (``pusula.judge``) runs between steps; this module only builds each pass's questions
and merges answers into training Examples.

    python -m pusula.label_real prepare  --inp clean.jsonl --out p1.jsonl
    python -m pusula.judge --inp p1.jsonl --out j1.jsonl --source q1 --policy --questions alan,<universal names>
    python -m pusula.label_real advance  --inp j1.jsonl --out p2.jsonl
    python -m pusula.judge --inp p2.jsonl --out j2.jsonl --source q2 --policy --questions alt_alan
    python -m pusula.label_real advance  --inp j2.jsonl --out p3.jsonl
    python -m pusula.judge --inp p3.jsonl --out j3.jsonl --source q3 --policy --questions niyet
    python -m pusula.label_real finalize --inp j3.jsonl --out train.jsonl --report report.json

    python -m pusula.label_real questions
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from .jsonl import read_jsonl, write_examples
from .schema import Example
from .taxonomy import Q_DOMAIN, Q_INTENT, Q_SUBDOMAIN, load_taxonomy

SOURCE_Q1, SOURCE_Q2, SOURCE_Q3 = "q1", "q2", "q3"
DEFAULT_SOURCES = (SOURCE_Q1, SOURCE_Q2, SOURCE_Q3)
FINAL = "final"
LABEL_SOURCE = "judge-hier"


def read_judged_examples(path: str | Path) -> list[Example]:
    """Load judged rows as validated Examples; a row that fails validation (an answer that is not an option key,
    which pusula.judge never writes) is loaded unvalidated so that advance/finalize can reject it with a reason."""
    out = []
    for row in read_jsonl(path):
        try:
            out.append(Example(**row))
        except ValidationError:
            out.append(Example.model_construct(**row))
    return out


def pass1_question_names() -> list[str]:
    tax = load_taxonomy()
    return [Q_DOMAIN, *tax.universal.keys()]


def _reject(ex: Example, reason: str) -> Example:
    meta = dict(ex.meta)
    meta["reason"] = reason
    return ex.model_copy(update={"meta": meta})


def _parse_sources(raw: str | None) -> tuple[str, str, str]:
    if not raw:
        return DEFAULT_SOURCES
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--sources must be three comma-separated names, e.g. q1,q2,q3")
    return parts[0], parts[1], parts[2]


def prepare(inp: str, out: str) -> dict[str, int]:
    tax = load_taxonomy()
    questions = {Q_DOMAIN: tax.domain_question(), **tax.universal}
    rows: list[Example] = []
    skipped = 0
    for row in read_jsonl(inp):
        state = (row.get("state") or "").strip()
        if not state:
            skipped += 1
            continue
        meta = dict(row.get("meta") or {})
        meta["label_source"] = LABEL_SOURCE
        rows.append(
            Example(id=row["id"], state=state, questions=questions, labels={}, meta=meta),
        )
    write_examples(out, rows)
    return {"written": len(rows), "skipped_empty_state": skipped}


def advance(
    inp: str,
    out: str,
    rejects: str,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> dict[str, int]:
    tax = load_taxonomy()
    s1, s2, _ = sources
    kept: list[Example] = []
    rejected: list[Example] = []
    counts = Counter(advanced_subdomain=0, advanced_intent=0, rejected=0)

    for ex in read_judged_examples(inp):
        if Q_SUBDOMAIN not in ex.questions:
            domain = ex.labels.get(s1, {}).get(Q_DOMAIN)
            if not domain or domain not in tax.domains:
                rejected.append(_reject(ex, "invalid_domain"))
                counts["rejected"] += 1
                continue
            qs = dict(ex.questions)
            qs[Q_SUBDOMAIN] = tax.subdomain_question(domain)
            kept.append(ex.model_copy(update={"questions": qs}))
            counts["advanced_subdomain"] += 1
        elif Q_INTENT not in ex.questions:
            domain = ex.labels.get(s1, {}).get(Q_DOMAIN)
            if not domain or domain not in tax.domains:
                rejected.append(_reject(ex, "invalid_domain"))
                counts["rejected"] += 1
                continue
            subdomain = ex.labels.get(s2, {}).get(Q_SUBDOMAIN)
            if not subdomain or subdomain not in tax.domains[domain].subdomains:
                rejected.append(_reject(ex, "invalid_subdomain"))
                counts["rejected"] += 1
                continue
            qs = dict(ex.questions)
            qs[Q_INTENT] = tax.intent_question(domain, subdomain)
            kept.append(ex.model_copy(update={"questions": qs}))
            counts["advanced_intent"] += 1
        else:
            rejected.append(_reject(ex, "already_complete"))
            counts["rejected"] += 1

    write_examples(out, kept)
    write_examples(rejects, rejected)
    return dict(counts)


def finalize(
    inp: str,
    out: str,
    rejects: str,
    report_path: str | None,
    sources: Sequence[str] = DEFAULT_SOURCES,
) -> dict:
    tax = load_taxonomy()
    s1, s2, s3 = sources
    kept: list[Example] = []
    rejected: list[Example] = []
    reject_reasons: Counter[str] = Counter()
    label_dist: dict[str, Counter[str]] = defaultdict(Counter)
    dropped_invalid_universal = 0

    examples = read_judged_examples(inp)
    for ex in examples:
        l1 = ex.labels.get(s1, {})
        l2 = ex.labels.get(s2, {})
        l3 = ex.labels.get(s3, {})
        domain = l1.get(Q_DOMAIN)
        subdomain = l2.get(Q_SUBDOMAIN)
        intent = l3.get(Q_INTENT)

        reason: str | None = None
        if not domain or domain not in tax.domains:
            reason = "invalid_domain"
        elif not subdomain or subdomain not in tax.domains[domain].subdomains:
            reason = "invalid_subdomain"
        elif not intent or intent not in tax.domains[domain].subdomains[subdomain].intents:
            reason = "invalid_intent"

        if reason:
            rejected.append(_reject(ex, reason))
            reject_reasons[reason] += 1
            continue

        path = tax.path(domain, subdomain, intent)
        merged = {Q_DOMAIN: domain, Q_SUBDOMAIN: subdomain, Q_INTENT: intent}
        for name, q in tax.universal.items():
            if name not in l1:
                continue
            key = l1[name]
            if key in q.option_keys():
                merged[name] = key
            else:
                dropped_invalid_universal += 1

        for qname, val in merged.items():
            label_dist[qname][val] += 1

        kept.append(
            ex.model_copy(
                update={
                    "questions": tax.questions_for(path),
                    "labels": {FINAL: merged},
                    "teacher": {},
                },
            ),
        )

    write_examples(out, kept)
    write_examples(rejects, rejected)

    report = {
        "input": len(examples),
        "kept": len(kept),
        "rejected": dict(sorted(reject_reasons.items())),
        "dropped_invalid_universal": dropped_invalid_universal,
        "labels": {q: dict(sorted(c.items())) for q, c in sorted(label_dist.items())},
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare")
    s.add_argument("--inp", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("advance")
    s.add_argument("--inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rejects", default=None, help="default: <out>.rejects.jsonl")
    s.add_argument("--sources", default=None)

    s = sub.add_parser("finalize")
    s.add_argument("--inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rejects", default=None)
    s.add_argument("--report", default=None)
    s.add_argument("--sources", default=None)

    sub.add_parser("questions")

    a = p.parse_args(argv)
    if a.cmd == "questions":
        print(",".join(pass1_question_names()))
        return

    sources = _parse_sources(getattr(a, "sources", None))
    if a.cmd == "prepare":
        print(json.dumps(prepare(a.inp, a.out), ensure_ascii=False))
    elif a.cmd == "advance":
        rejects = a.rejects or f"{a.out}.rejects.jsonl"
        print(json.dumps(advance(a.inp, a.out, rejects, sources), ensure_ascii=False))
    else:
        rejects = a.rejects or f"{a.out}.rejects.jsonl"
        report = finalize(a.inp, a.out, rejects, a.report, sources)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
