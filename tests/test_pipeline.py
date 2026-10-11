"""Data pipeline logic: generation specs/parsing, judge lettering, consensus, gold round-trip."""
import json

import pytest

from pusula import dataset, gold, judge
from pusula.generate import ADHOC_QUESTION, LENGTHS, SHORT_STYLES, make_specs, output_schema, to_example
from pusula.jsonl import read_examples, write_examples


def test_specs_cover_every_intent_once_per_cycle(tax):
    n = len(list(tax.intent_paths()))
    specs = make_specs(tax, 0, n, seed=1)
    assert len({str(s.path) for s in specs}) == n
    short = {l for l, _ in LENGTHS[:2]}
    assert all(s.length in short for s in specs if s.style in SHORT_STYLES)


def test_output_schema_adhoc_toggle(tax):
    assert ADHOC_QUESTION not in output_schema(tax, False)["properties"]
    assert ADHOC_QUESTION in output_schema(tax, True)["required"]


def _llm_output(adhoc_answer="hizli"):
    return {
        "senaryo": "...", "state": "Siparişim üç gündür kargoda görünmüyor, ne zaman gelir?",
        "aciliyet": 1, "eylem_gerekli": True, "insana_devret": False, "guvenlik": "guvenli",
        "duygu": "endiseli", "kisisel_veri": False,
        ADHOC_QUESTION: {"soru": "Kargo firmasına hangi kanaldan yazılsın?", "cevap": adhoc_answer,
                         "secenekler": [{"anahtar": "hizli", "aciklama": "Hızlı kanal"},
                                        {"anahtar": "normal", "aciklama": "Normal kanal"},
                                        {"anahtar": "Bad Key!", "aciklama": "geçersiz"},
                                        {"anahtar": "yok", "aciklama": "Yazma"}]},
    }


def test_to_example_parses_llm_output(tax):
    spec = make_specs(tax, 0, 1, seed=3)[0]
    spec.adhoc = True
    ex = to_example(tax, spec, _llm_output(), "qwen")
    gen = ex.labels["generator"]
    assert gen["aciliyet"] == "1" and gen["eylem_gerekli"] == "true"
    assert list(ex.questions[ADHOC_QUESTION].criteria) == ["hizli", "normal", "yok"]
    assert gen[ADHOC_QUESTION] == "hizli"
    # an answer that is not among the options drops the ad-hoc question
    ex2 = to_example(tax, spec, _llm_output("baska"), "qwen")
    assert ADHOC_QUESTION not in ex2.questions


def test_judge_lettering_roundtrip(example):
    order = judge.lettered(example, seed=5)
    assert order["aciliyet"] == example.questions["aciliyet"].option_keys()  # ordinal order kept
    letters = {name: judge.LETTERS[0] for name in order}
    decoded = judge.decode(order, letters)
    assert all(decoded[n] == order[n][0] for n in order)
    schema = judge.answer_schema(order)
    assert set(schema["required"]) == set(example.questions)
    assert "Durum:" in judge.build_prompt(example, order)


@pytest.mark.parametrize("qtype,votes,expected", [
    ("choice", ["a", "a", "b"], "a"),
    ("choice", ["a", "b", "c"], None),
    ("choice", ["a"], None),
    ("score", ["1", "2", "2"], "2"),
    ("score", ["0", "3", "3"], "3"),
    ("score", ["0", "2", "3"], None),
])
def test_consensus_rules(qtype, votes, expected):
    from pusula.schema import Question
    q = (Question(type="choice", instructions="x", criteria={"a": None, "b": None, "c": None})
         if qtype == "choice" else Question(type="score", instructions="x", criteria=["0", "1", "2", "3"]))
    assert dataset.consensus(q, votes) == expected


def _labeled(example, n=40):
    rows = []
    for i in range(n):
        ex = example.model_copy(deep=True)
        ex.id = f"ex{i}"
        gen = ex.labels["generator"]
        ex.labels["judge"] = dict(gen)
        d3 = dict(gen)
        if i % 4 == 0:  # d3 disagrees on duygu -> label stays (2 of 3), soft target dropped
            d3["duygu"] = "ofkeli"
        ex.labels["d3"] = d3
        ex.teacher = {name: [1.0 if k == d3[name] else 0.0 for k in q.option_keys()]
                      for name, q in ex.questions.items()}
        if i % 5 == 0:  # three different answers -> contested
            ex.labels["judge"]["guvenlik"] = "spam_anlamsiz"
            ex.labels["d3"]["guvenlik"] = "jailbreak_rol_yapma"
        rows.append(ex)
    return rows


def test_build_and_gold_roundtrip(tmp_path, example):
    write_examples(tmp_path / "labeled.jsonl", _labeled(example))
    report = dataset.build(str(tmp_path / "labeled.jsonl"), str(tmp_path / "final"), val=0.2, test=0.3)
    assert sum(report["splits"].values()) == 40
    assert report["questions"]["guvenlik"]["contested"] == 8
    rows = read_examples(tmp_path / "final" / "train.jsonl") + read_examples(tmp_path / "final" / "test.jsonl")
    ex0 = next(r for r in rows if r.id == "ex0") if any(r.id == "ex0" for r in rows) else None
    if ex0:
        assert "duygu" not in ex0.teacher and ex0.labels["final"]["duygu"] == "endiseli"

    n = gold.export(str(tmp_path / "final"), 10, str(tmp_path / "gold"))
    blind = json.loads((tmp_path / "gold" / "blind.json").read_text(encoding="utf-8"))
    assert n == len(blind) and all("labels" not in b and "suggestion" not in b for b in blind)
    cands = read_examples(tmp_path / "gold" / "candidates.jsonl")
    labels = [{"id": c.id, "labels": c.labels["generator"], "confidence": "high"} for c in cands]
    labels[0]["drop"] = True
    labels[1]["labels"] = {**labels[1]["labels"], "alan": "yok"}  # invalid path -> skipped
    (tmp_path / "gold" / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    m = gold.import_labels(str(tmp_path / "gold" / "candidates.jsonl"), str(tmp_path / "gold" / "labels.json"),
                           str(tmp_path / "gold" / "gold.jsonl"))
    assert m == n - 2
    g = read_examples(tmp_path / "gold" / "gold.jsonl")[0]
    assert g.labels["gold"] and g.meta["gold_confidence"] == "high"
    assert g.labels["generator"]["alan"] == g.labels["gold"]["alan"]
