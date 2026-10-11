"""Stage 6: evaluate a trained model on labeled splits and measure latency.

For each file, the reference label source is ``gold`` when present (gold set), else ``final``.
On the gold set the other sources (generator, judge, d3) are scored too, so the student can be
compared with its teacher.

    python -m pusula.evaluate --model runs/full/model/best --data runs/full/final/test.jsonl \
        runs/full/gold/gold.jsonl --out runs/full/reports
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .jsonl import read_examples
from .llm import console
from .schema import Example

OTHER_SOURCES = ("generator", "judge", "d3")


def _ece(conf: list[float], ok: list[int], bins: int = 15) -> float:
    total, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            total += len(sel) / n * abs(sum(ok[i] for i in sel) / len(sel) - sum(conf[i] for i in sel) / len(sel))
    return total


def evaluate_file(engine, examples: list[Example], batch_size: int = 16) -> dict:
    ref_source = "gold" if any("gold" in ex.labels for ex in examples) else "final"
    per_q = defaultdict(lambda: {"n": 0, "ok": 0, "brier": 0.0, "abs_err": 0.0})
    others = {s: defaultdict(lambda: [0, 0]) for s in OTHER_SOURCES}
    conf, oks = [], []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        reqs = [(ex.state, {n: ex.questions[n] for n in ex.labels.get(ref_source, {})}) for ex in chunk]
        answers = engine.decide_many(reqs)
        for ex, ans in zip(chunk, answers):
            for name, ref in ex.labels.get(ref_source, {}).items():
                q = ex.questions[name]
                a = ans[name]
                keys = q.option_keys()
                if q.type == "noul":
                    probs = [1 - a["noul"], a["noul"]]
                else:
                    probs = [a["probabilities"][k] for k in keys]
                pred = keys[max(range(len(probs)), key=probs.__getitem__)]
                st = per_q[name]
                st["n"] += 1
                st["ok"] += pred == ref
                st["brier"] += sum((p - (k == ref)) ** 2 for p, k in zip(probs, keys))
                if q.type == "score":
                    st["abs_err"] += abs(a["score"] - int(ref))
                conf.append(max(probs))
                oks.append(int(pred == ref))
                for s in OTHER_SOURCES:
                    v = ex.labels.get(s, {}).get(name)
                    if v is not None:
                        others[s][name][0] += 1
                        others[s][name][1] += v == ref
    questions = {}
    for name, st in sorted(per_q.items()):
        row = {"n": st["n"], "acc": round(st["ok"] / st["n"], 4), "brier": round(st["brier"] / st["n"], 4)}
        if st["abs_err"]:
            row["score_mae"] = round(st["abs_err"] / st["n"], 4)
        questions[name] = row
    report = {
        "reference": ref_source,
        "examples": len(examples),
        "macro_acc": round(sum(r["acc"] for r in questions.values()) / max(len(questions), 1), 4),
        "ece": round(_ece(conf, oks), 4),
        "questions": questions,
    }
    if ref_source == "gold":
        report["sources"] = {
            s: {n: round(ok / tot, 4) for n, (tot, ok) in sorted(v.items()) if tot}
            for s, v in others.items()
        }
    return report


def measure_latency(engine, runs: int = 200) -> dict:
    """Typical request: a short message, taxonomy domain question (20 options) + universal questions."""
    from .taxonomy import load_taxonomy

    tax = load_taxonomy()
    qs = {"alan": tax.domain_question(), **tax.universal}
    state = "Kartımdan bilgim dışında 4.500 TL çekilmiş, hemen bloke edilmesini istiyorum!"
    for _ in range(10):
        engine.decide(state, qs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        engine.decide(state, qs)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    pairs = sum(len(q.option_keys()) for q in qs.values())
    return {"pairs_per_request": pairs, "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(times[int(0.95 * len(times)) - 1], 2), "runs": runs}


def to_markdown(results: dict) -> str:
    lines = ["# Pusula değerlendirme raporu", ""]
    if "latency" in results:
        lat = results["latency"]
        lines += [f"Gecikme ({lat['pairs_per_request']} seçenek çifti/istek): p50 {lat['p50_ms']} ms, "
                  f"p95 {lat['p95_ms']} ms", ""]
    for name, rep in results["files"].items():
        lines += [f"## {name}", "", f"Referans: `{rep['reference']}` · örnek: {rep['examples']} · "
                  f"makro doğruluk: **{rep['macro_acc']:.2%}** · ECE: {rep['ece']:.4f}", "",
                  "| soru | n | doğruluk | brier |", "|---|---:|---:|---:|"]
        for q, r in rep["questions"].items():
            lines.append(f"| {q} | {r['n']} | {r['acc']:.2%} | {r['brier']:.4f} |")
        if "sources" in rep:
            lines += ["", "Diğer kaynakların altın sete göre doğruluğu:", ""]
            names = sorted(rep["questions"])
            lines += ["| kaynak | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
            for s, accs in rep["sources"].items():
                lines.append(f"| {s} | " + " | ".join(f"{accs[n]:.2%}" if n in accs else "-" for n in names) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    from .inference import DecisionEngine

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-latency", action="store_true")
    a = p.parse_args(argv)
    engine = DecisionEngine.load(a.model)
    results: dict = {"files": {}}
    for path in a.data:
        if not Path(path).exists():
            console.print(f"[yellow]{path} yok, atlandı[/yellow]")
            continue
        results["files"][Path(path).name] = evaluate_file(engine, read_examples(path))
    if not a.no_latency:
        results["latency"] = measure_latency(engine)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "eval.md").write_text(to_markdown(results), encoding="utf-8")
    console.print(to_markdown(results))


if __name__ == "__main__":
    main()
