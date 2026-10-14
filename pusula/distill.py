"""Pusula Nano from Pusula: knowledge-distillation data.

    # 1. sample training examples, add a wide many-option intent question (30-60 options) so the small model
    #    learns to pick among many options, copy val/test with the same extra question
    python -m pusula.distill prepare --data runs/full/curated --out runs/full/distill --n 3000
    # 2. Pusula labels every question (soft targets)
    python -m pusula.teacher --d3 ~/d3 --adapter runs/full/pusula-d3/best --source pusula --precision bf16 \
        --inp runs/full/distill/train_in.jsonl --out runs/full/distill/train_labeled.jsonl
    # 3. keep Pusula's distribution only where it agrees with the consensus label
    python -m pusula.distill finalize --out runs/full/distill
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .jsonl import read_examples, write_examples
from .llm import console
from .schema import Example, Question, argmax_key
from .taxonomy import Q_INTENT, load_taxonomy

WIDE_QUESTION = "niyet_genis"


def add_wide_intent(ex: Example, rng: random.Random, lo: int = 30, hi: int = 60) -> bool:
    """True intent among `lo..hi` options drawn from the whole taxonomy (shuffled)."""
    true = ex.labels.get("final", {}).get(Q_INTENT)
    if not true:
        return False
    descs = {p.intent: d for p, d in _intent_descriptions()}
    others = [k for k in descs if k != true]
    keys = rng.sample(others, rng.randint(lo, hi) - 1) + [true]
    rng.shuffle(keys)
    ex.questions[WIDE_QUESTION] = Question(type="choice", instructions="Kullanıcının asıl niyeti aşağıdakilerden hangisi?",
                                           criteria={k: descs[k] for k in keys})
    ex.labels["final"][WIDE_QUESTION] = true
    return True


_DESCS = None


def _intent_descriptions():
    global _DESCS
    if _DESCS is None:
        tax = load_taxonomy()
        _DESCS = [(p, tax.domains[p.domain].subdomains[p.subdomain].intents[p.intent]) for p in tax.intent_paths()]
    return _DESCS


def prepare(data: str, out: str, n: int, seed: int = 23) -> None:
    """train_in.jsonl: the n examples Pusula labels; train_rest.jsonl: the others (hard labels only)."""
    rng = random.Random(seed)
    src, dst = Path(data), Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    train = read_examples(src / "train.jsonl")
    rng.shuffle(train)
    for split, rows in (("train_in", train[:n]), ("train_rest", train[n:]), ("val", read_examples(src / "val.jsonl")),
                        ("test", read_examples(src / "test.jsonl"))):
        added = sum(add_wide_intent(ex, rng) for ex in rows)
        write_examples(dst / f"{split}.jsonl", rows)
        console.print(f"{split}: {len(rows)} örnek, {added} geniş niyet sorusu")


def finalize(out: str, source: str = "pusula", seed: int = 29) -> None:
    """full/train.jsonl = the Pusula-labeled examples + all hard-label ones (Nano's training data)."""
    dst = Path(out)
    rows = read_examples(dst / "train_labeled.jsonl")
    kept = dropped = 0
    for ex in rows:
        for name in list(ex.teacher):
            final = ex.labels.get("final", {}).get(name)
            q = ex.questions.get(name)
            if final is None or q is None or argmax_key(q, ex.teacher[name]) != final:
                ex.teacher.pop(name)
                dropped += 1
            else:
                kept += 1
    rest = read_examples(dst / "train_rest.jsonl")
    random.Random(seed).shuffle(rest)
    full = dst / "full"
    full.mkdir(exist_ok=True)
    write_examples(full / "train.jsonl", rows + rest)
    for split in ("val", "test"):
        write_examples(full / f"{split}.jsonl", read_examples(dst / f"{split}.jsonl"))
    console.print(f"[green]{len(rows)} örnek Pusula etiketli: {kept} soru dağılımıyla, {dropped} soru yalnız etiketle; "
                  f"Nano {len(rows) + len(rest)} örnek[/green]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=3000)
    b = sub.add_parser("finalize")
    b.add_argument("--out", required=True)
    b.add_argument("--source", default="pusula")
    args = p.parse_args(argv)
    prepare(args.data, args.out, args.n) if args.cmd == "prepare" else finalize(args.out, args.source)


if __name__ == "__main__":
    main()
