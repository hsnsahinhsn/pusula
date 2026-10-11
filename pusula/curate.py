"""TabiBERT-assisted curation of the Turkish training data before d3 fine-tuning.

1. Semantic de-duplication: states whose TabiBERT embeddings are near-identical (cosine >= --dup)
   are dropped, keeping the first.
2. Label-noise detection: the TabiBERT cross-encoder is trained on k-1 folds and scores the held-out
   fold; a label the model rejects with high confidence (label prob < --noise-label while its own top
   choice has prob > --noise-top) is removed from `labels.final` (recorded in `meta.noisy`).
3. Hard distractors: a new choice question `niyet_zor` pairs the true intent with the intents from
   OTHER subdomains that TabiBERT finds most similar to the message, so d3 learns fine distinctions
   and unfamiliar option sets.

    python -m pusula.curate --data runs/full/final --out runs/full/curated
"""
from __future__ import annotations

import argparse
import json
import random
from argparse import Namespace
from pathlib import Path

from .jsonl import read_examples, write_examples
from .llm import console
from .model import BASE_MODEL
from .schema import Example, Question
from .taxonomy import Q_INTENT, load_taxonomy

HARD_QUESTION = "niyet_zor"


def embed(texts: list[str], model_name: str = BASE_MODEL, batch: int = 64, max_length: int = 512):
    """L2-normalized mean-pooled TabiBERT embeddings [N, H] (torch, on the model's device)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1))
    return torch.cat(out)


def dedup(examples: list[Example], emb, threshold: float) -> list[int]:
    """Indices to keep: an example is dropped if it is >= threshold similar to an earlier kept one."""
    import torch

    kept = torch.empty((len(examples), emb.shape[1]), dtype=emb.dtype, device=emb.device)
    keep: list[int] = []
    for i in range(len(examples)):
        v = emb[i]
        if keep and (kept[: len(keep)] @ v).max().item() >= threshold:
            continue
        kept[len(keep)] = v
        keep.append(i)
    return keep


def add_hard_question(examples: list[Example], emb, k: int = 5) -> int:
    """Adds `niyet_zor` (true intent + k most similar intents from other subdomains) where niyet is labeled."""
    tax = load_taxonomy()
    paths = list(tax.intent_paths())
    descs = [tax.domains[p.domain].subdomains[p.subdomain].intents[p.intent] for p in paths]
    intent_emb = embed(descs)
    sims = emb @ intent_emb.T
    added = 0
    for i, ex in enumerate(examples):
        true = ex.labels.get("final", {}).get(Q_INTENT)
        if not true:
            continue
        sub = ex.labels["final"].get("alt_alan")
        order = sims[i].argsort(descending=True).tolist()
        options = {}
        for j in order:
            p = paths[j]
            if p.intent == true and p.subdomain == sub:
                continue
            if p.subdomain == sub or p.intent in options or p.intent == true:
                continue
            options[p.intent] = descs[j]
            if len(options) == k:
                break
        true_desc = next(descs[j] for j, p in enumerate(paths) if p.intent == true and p.subdomain == sub)
        options[true] = true_desc
        keys = list(options)
        random.Random(ex.id).shuffle(keys)
        ex.questions[HARD_QUESTION] = Question(
            type="choice", instructions="Kullanıcının asıl niyeti aşağıdakilerden hangisi?",
            criteria={key: options[key] for key in keys})
        ex.labels["final"][HARD_QUESTION] = true
        added += 1
    return added


def flag_noise(examples: list[Example], folds: int, label_thr: float, top_thr: float, epochs: int,
               workdir: Path) -> int:
    """k-fold TabiBERT cross-encoder; removes confidently rejected labels from labels.final."""
    import torch
    from transformers import AutoTokenizer

    from . import train as T
    from .model import PairEncoder, PusulaModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = list(range(len(examples)))
    random.Random(5).shuffle(idx)
    flagged = 0
    for f in range(folds):
        held = set(idx[f::folds])
        train_ex = [ex for i, ex in enumerate(examples) if i not in held]
        held_ex = [examples[i] for i in sorted(held)]
        fold_dir = workdir / f"fold{f}"
        (fold_dir / "data").mkdir(parents=True, exist_ok=True)
        write_examples(fold_dir / "data" / "train.jsonl", train_ex)
        write_examples(fold_dir / "data" / "val.jsonl", held_ex[: max(50, len(held_ex) // 20)])
        T.train(Namespace(data=str(fold_dir / "data"), out=str(fold_dir / "model"), base_model=BASE_MODEL,
                          max_length=512, epochs=epochs, lr=3e-5, head_lr=1e-4, weight_decay=0.01, warmup=0.06,
                          alpha=1.0, token_budget=65536, max_train_options=8, log_every=200, seed=7 + f))
        model, tok, _ = PusulaModel.load(fold_dir / "model" / "best", device=device)
        enc = PairEncoder(tok or AutoTokenizer.from_pretrained(BASE_MODEL), max_length=512)
        data = T.Encoded(held_ex, enc)
        for it, lg in T.predict(model, enc, data, device, 65536):
            p = torch.softmax(lg, dim=-1)
            if p[it.label].item() < label_thr and p.max().item() > top_thr:
                ex = held_ex[it.state]
                # routing questions depend on the level above: drop the dependent labels too
                dependents = {"alan": ["alan", "alt_alan", "niyet"], "alt_alan": ["alt_alan", "niyet"]}
                for name in dependents.get(it.name, [it.name]):
                    if ex.labels["final"].pop(name, None) is not None:
                        ex.meta.setdefault("noisy", []).append(name)
                        flagged += 1
        del model
        torch.cuda.empty_cache()
    return flagged


def curate(a) -> dict:
    data, out = Path(a.data), Path(a.out)
    report: dict = {}
    for split in ("train", "val", "test"):
        examples = read_examples(data / f"{split}.jsonl")
        n0 = len(examples)
        emb = embed([ex.state for ex in examples])
        keep = dedup(examples, emb, a.dup) if split == "train" else list(range(n0))
        examples = [examples[i] for i in keep]
        emb = emb[keep]
        flagged = 0
        if split == "train" and a.folds > 1:
            flagged = flag_noise(examples, a.folds, a.noise_label, a.noise_top, a.noise_epochs, out / "noise")
        hard = add_hard_question(examples, emb, a.hard_k)
        examples = [ex for ex in examples if ex.labels.get("final")]
        write_examples(out / f"{split}.jsonl", examples)
        report[split] = {"in": n0, "after_dedup": len(keep), "noisy_labels_removed": flagged,
                         "hard_questions": hard, "out": len(examples)}
        console.print(f"[green]{split}: {report[split]}[/green]")
    (out / "curate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="consensus split directory (train/val/test.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--dup", type=float, default=0.95, help="cosine similarity treated as duplicate")
    p.add_argument("--folds", type=int, default=3, help="noise-detection folds (1 disables)")
    p.add_argument("--noise-epochs", type=int, default=1)
    p.add_argument("--noise-label", type=float, default=0.05)
    p.add_argument("--noise-top", type=float, default=0.9)
    p.add_argument("--hard-k", type=int, default=5)
    curate(p.parse_args(argv))


if __name__ == "__main__":
    main()
