"""Post-hoc calibration of Pusula's built-in decision questions on an independently labeled set.

Each calibrated question gets one additive bias per option on the log-probabilities (for a yes/no question this
is a threshold shift), fitted by minimizing the negative log-likelihood on a held-out labeled set. The bias is a
handful of numbers, so a few hundred labeled examples are enough, and it is checked on a separate set.

    # probabilities of a model on a labeled file
    python -m pusula.calibrate predict --d3 ~/d3 --adapter pusula --data gold.jsonl --out gold_preds.jsonl
    # fit on one set, report the effect on another
    python -m pusula.calibrate fit --preds gold_preds.jsonl --out pusula/calibration.json \
        --questions aciliyet,insana_devret,eylem_gerekli,kisisel_veri
    python -m pusula.calibrate score --preds bench_preds.jsonl --calibration pusula/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .jsonl import read_examples
from .llm import console

FORMAT = "pusula-calibration-v1"


def predict(d3: str, adapter: Optional[str], data: str, out: str, token_budget: int = 16384) -> int:
    import torch

    from .d3model import D3Decider, render

    decider = D3Decider.load(d3, adapter=adapter, load_in_4bit=False)
    decider.backbone.eval()
    examples = read_examples(data)
    source = "gold" if any("gold" in ex.labels for ex in examples) else "final"
    jobs = [(ex, name) for ex in examples for name in ex.labels.get(source, {})]
    ids = decider.encode([render(decider.tokenizer, ex.state, ex.questions[n], decider.codes) for ex, n in jobs])
    order = sorted(range(len(jobs)), key=lambda i: len(ids[i]))
    rows: list[dict] = [None] * len(jobs)  # type: ignore[list-item]
    batch: list[int] = []

    def flush():
        if not batch:
            return
        counts = [len(jobs[i][0].questions[jobs[i][1]].option_keys()) for i in batch]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            probs = torch.softmax(decider.logits([ids[i] for i in batch], counts), dim=-1).float().cpu()
        for r, i in enumerate(batch):
            ex, name = jobs[i]
            keys = ex.questions[name].option_keys()
            rows[i] = {"id": ex.id, "name": name, "probs": probs[r, : counts[r]].tolist(),
                       "label": keys.index(ex.labels[source][name])}
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
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    console.print(f"[green]{len(rows)} tahmin → {out} (referans: {source})[/green]")
    return len(rows)


def _load(preds: str) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for line in open(preds, encoding="utf-8"):
        row = json.loads(line)
        by_name[row["name"]].append(row)
    return by_name


def apply_bias(probs: list[float], bias: list[float]) -> list[float]:
    z = [math.log(max(p, 1e-12)) + b for p, b in zip(probs, bias)]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def fit(preds: str, out: str, questions: list[str], l2: float = 0.01) -> dict:
    import torch

    by_name = _load(preds)
    calib = {"format_id": FORMAT, "questions": {}}
    for name in questions:
        rows = [r for r in by_name.get(name, [])]
        if not rows:
            continue
        k = len(rows[0]["probs"])
        rows = [r for r in rows if len(r["probs"]) == k]
        logp = torch.tensor([[math.log(max(p, 1e-12)) for p in r["probs"]] for r in rows])
        y = torch.tensor([r["label"] for r in rows])
        b = torch.zeros(k, requires_grad=True)
        opt = torch.optim.LBFGS([b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logp + b, y) + l2 * (b ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        bias = (b - b.mean()).detach().tolist()  # softmax is shift invariant: center for readability
        before = (logp.argmax(1) == y).float().mean().item()
        after = ((logp + torch.tensor(bias)).argmax(1) == y).float().mean().item()
        calib["questions"][name] = {"options": k, "bias": [round(v, 4) for v in bias], "fit_n": len(rows),
                                    "fit_acc_before": round(before, 4), "fit_acc_after": round(after, 4)}
        console.print(f"{name}: {len(rows)} örnek, uyum {before:.3f} → {after:.3f}, bias {[round(v, 2) for v in bias]}")
    Path(out).write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    return calib


def score(preds: str, calibration: str) -> dict:
    calib = json.loads(Path(calibration).read_text(encoding="utf-8"))["questions"]
    by_name = _load(preds)
    report = {}
    for name, rows in sorted(by_name.items()):
        acc0 = sum(max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["label"] for r in rows) / len(rows)
        c = calib.get(name)
        if c:  # rows whose option count differs from the fitted one keep their uncalibrated prediction
            acc1 = sum(max(range(len(r["probs"])), key=(apply_bias(r["probs"], c["bias"]) if len(r["probs"]) == c["options"]
                                                        else r["probs"]).__getitem__) == r["label"] for r in rows) / len(rows)
        else:
            acc1 = acc0
        report[name] = {"n": len(rows), "acc": round(acc0, 4), "acc_calibrated": round(acc1, 4)}
    macro0 = sum(v["acc"] for v in report.values()) / len(report)
    macro1 = sum(v["acc_calibrated"] for v in report.values()) / len(report)
    report["_macro"] = {"acc": round(macro0, 4), "acc_calibrated": round(macro1, 4)}
    console.print_json(data=report)
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("predict")
    a.add_argument("--d3", required=True)
    a.add_argument("--adapter", default=None)
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("fit")
    b.add_argument("--preds", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--questions", default="aciliyet,insana_devret,eylem_gerekli,kisisel_veri")
    b.add_argument("--l2", type=float, default=0.01)
    c = sub.add_parser("score")
    c.add_argument("--preds", required=True)
    c.add_argument("--calibration", required=True)
    args = p.parse_args(argv)
    if args.cmd == "predict":
        predict(args.d3, args.adapter, args.data, args.out)
    elif args.cmd == "fit":
        fit(args.preds, args.out, args.questions.split(","), args.l2)
    else:
        score(args.preds, args.calibration)


if __name__ == "__main__":
    main()
