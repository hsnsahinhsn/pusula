import json
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from pusula.jsonl import read_examples, write_examples
from pusula.label_real import (
    SOURCE_Q1,
    SOURCE_Q2,
    SOURCE_Q3,
    advance,
    finalize,
    pass1_question_names,
    prepare,
    read_judged_examples,
)
from pusula.schema import Example
from pusula.taxonomy import Q_DOMAIN, Q_INTENT, Q_SUBDOMAIN, load_taxonomy


def _path_keys():
    tax = load_taxonomy()
    domain = next(iter(tax.domains))
    subdomain = next(iter(tax.domains[domain].subdomains))
    intent = next(iter(tax.domains[domain].subdomains[subdomain].intents))
    other_domain = next(d for d in tax.domains if d != domain)
    wrong_sub = next(iter(tax.domains[other_domain].subdomains))
    return tax, domain, subdomain, intent, other_domain, wrong_sub


def fake_judge(examples: list[Example], source: str, answers_fn) -> list[Example]:
    out = []
    for ex in examples:
        ans = answers_fn(ex)
        labels = {k: dict(v) for k, v in ex.labels.items()}
        labels[source] = ans
        try:
            out.append(ex.model_copy(update={"labels": labels}))
        except ValidationError:
            data = ex.model_dump()
            data["labels"] = labels
            out.append(Example.model_construct(**data))
    return out


def _universal_answers(tax, overrides: dict | None = None) -> dict[str, str]:
    ans = {}
    for name, q in tax.universal.items():
        ans[name] = q.option_keys()[0]
    if overrides:
        ans.update(overrides)
    return ans


def test_prepare_and_questions_cli(tmp_path):
    tax = load_taxonomy()
    inp = tmp_path / "clean.jsonl"
    inp.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "state": "Merhaba, yardım lazım.", "meta": {"k": 1}}),
                json.dumps({"id": "b", "state": "   "}),
                json.dumps({"id": "c", "state": "İkinci mesaj."}),
            ]
        ),
        encoding="utf-8",
    )
    stats = prepare(str(inp), str(tmp_path / "p1.jsonl"))
    assert stats == {"written": 2, "skipped_empty_state": 1}
    rows = read_examples(tmp_path / "p1.jsonl")
    assert {r.id for r in rows} == {"a", "c"}
    for ex in rows:
        assert set(ex.questions) == {Q_DOMAIN, *tax.universal}
        assert ex.labels == {}
        assert ex.meta["label_source"] == "judge-hier"
    assert next(r for r in rows if r.id == "a").meta.get("k") == 1

    names = pass1_question_names()
    assert names[0] == Q_DOMAIN and names[1:] == list(tax.universal.keys())
    out = subprocess.check_output([sys.executable, "-m", "pusula.label_real", "questions"], text=True).strip()
    assert out == ",".join(names)


def test_happy_path_two_messages(tmp_path):
    tax, domain, subdomain, intent, _, _ = _path_keys()
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        "\n".join(
            json.dumps({"id": i, "state": f"Örnek mesaj {i}."})
            for i in ("m1", "m2")
        ),
        encoding="utf-8",
    )
    prepare(str(clean), str(tmp_path / "p1.jsonl"))
    p1 = read_examples(tmp_path / "p1.jsonl")

    def q1_answers(_ex):
        return {Q_DOMAIN: domain, **_universal_answers(tax)}

    j1 = fake_judge(p1, SOURCE_Q1, q1_answers)
    write_examples(tmp_path / "j1.jsonl", j1)
    advance(str(tmp_path / "j1.jsonl"), str(tmp_path / "p2.jsonl"), str(tmp_path / "p2.rejects.jsonl"))
    p2 = read_examples(tmp_path / "p2.jsonl")
    assert Q_SUBDOMAIN in p2[0].questions

    j2 = fake_judge(p2, SOURCE_Q2, lambda _ex: {Q_SUBDOMAIN: subdomain})
    write_examples(tmp_path / "j2.jsonl", j2)
    advance(str(tmp_path / "j2.jsonl"), str(tmp_path / "p3.jsonl"), str(tmp_path / "p3.rejects.jsonl"))
    p3 = read_examples(tmp_path / "p3.jsonl")
    assert Q_INTENT in p3[0].questions

    j3 = fake_judge(p3, SOURCE_Q3, lambda _ex: {Q_INTENT: intent})
    write_examples(tmp_path / "j3.jsonl", j3)
    report = finalize(
        str(tmp_path / "j3.jsonl"),
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "fin.rejects.jsonl"),
        str(tmp_path / "report.json"),
    )
    assert report["input"] == 2 and report["kept"] == 2 and report["rejected"] == {}
    path = tax.path(domain, subdomain, intent)
    want_qs = tax.questions_for(path)
    for ex in read_examples(tmp_path / "train.jsonl"):
        assert ex.questions == want_qs
        assert set(ex.labels) == {"final"}
        for name in want_qs:
            assert name in ex.labels["final"]
        assert ex.labels["final"][Q_DOMAIN] == domain
        assert ex.teacher == {}


def test_advance_rejects_invalid_domain(tmp_path):
    tax, domain, subdomain, intent, _, _ = _path_keys()
    prepare(str(_write_clean(tmp_path, "x")), str(tmp_path / "p1.jsonl"))
    ex = read_examples(tmp_path / "p1.jsonl")[0]
    j1 = fake_judge([ex], SOURCE_Q1, lambda _e: {Q_DOMAIN: "not_a_domain", **_universal_answers(tax)})
    write_examples(tmp_path / "j1.jsonl", j1)
    counts = advance(str(tmp_path / "j1.jsonl"), str(tmp_path / "p2.jsonl"), str(tmp_path / "r.jsonl"))
    assert counts["rejected"] == 1 and counts["advanced_subdomain"] == 0
    assert read_examples(tmp_path / "p2.jsonl") == []
    rej = read_judged_examples(tmp_path / "r.jsonl")
    assert rej[0].meta["reason"] == "invalid_domain"


def test_advance_rejects_wrong_subdomain(tmp_path):
    tax, domain, subdomain, intent, other_domain, wrong_sub = _path_keys()
    prepare(str(_write_clean(tmp_path, "y")), str(tmp_path / "p1.jsonl"))
    p1 = read_examples(tmp_path / "p1.jsonl")
    j1 = fake_judge(p1, SOURCE_Q1, lambda _e: {Q_DOMAIN: domain, **_universal_answers(tax)})
    write_examples(tmp_path / "j1.jsonl", j1)
    advance(str(tmp_path / "j1.jsonl"), str(tmp_path / "p2.jsonl"), str(tmp_path / "r1.jsonl"))
    p2 = read_examples(tmp_path / "p2.jsonl")
    j2 = fake_judge(p2, SOURCE_Q2, lambda _e: {Q_SUBDOMAIN: wrong_sub})
    write_examples(tmp_path / "j2.jsonl", j2)
    counts = advance(str(tmp_path / "j2.jsonl"), str(tmp_path / "p3.jsonl"), str(tmp_path / "r2.jsonl"))
    assert counts["rejected"] == 1
    assert read_judged_examples(tmp_path / "r2.jsonl")[0].meta["reason"] == "invalid_subdomain"


def test_finalize_rejects_invalid_intent(tmp_path):
    tax, domain, subdomain, intent, _, _ = _path_keys()
    bad_intent = "invalid_intent_key_xyz"
    prepare(str(_write_clean(tmp_path, "z")), str(tmp_path / "p1.jsonl"))
    p1 = read_examples(tmp_path / "p1.jsonl")
    j1 = fake_judge(p1, SOURCE_Q1, lambda _e: {Q_DOMAIN: domain, **_universal_answers(tax)})
    write_examples(tmp_path / "j1.jsonl", j1)
    advance(str(tmp_path / "j1.jsonl"), str(tmp_path / "p2.jsonl"), str(tmp_path / "r1.jsonl"))
    p2 = read_examples(tmp_path / "p2.jsonl")
    j2 = fake_judge(p2, SOURCE_Q2, lambda _e: {Q_SUBDOMAIN: subdomain})
    write_examples(tmp_path / "j2.jsonl", j2)
    advance(str(tmp_path / "j2.jsonl"), str(tmp_path / "p3.jsonl"), str(tmp_path / "r2.jsonl"))
    p3 = read_examples(tmp_path / "p3.jsonl")
    j3 = fake_judge(p3, SOURCE_Q3, lambda _e: {Q_INTENT: bad_intent})
    write_examples(tmp_path / "j3.jsonl", j3)
    report = finalize(
        str(tmp_path / "j3.jsonl"),
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "fin.rejects.jsonl"),
        None,
    )
    assert report["kept"] == 0 and report["rejected"] == {"invalid_intent": 1}
    assert read_judged_examples(tmp_path / "fin.rejects.jsonl")[0].meta["reason"] == "invalid_intent"


def test_finalize_drops_invalid_universal_and_report(tmp_path):
    tax, domain, subdomain, intent, _, _ = _path_keys()
    uni_name = next(iter(tax.universal))
    bad_key = "not_a_valid_option_key"
    prepare(str(_write_clean(tmp_path, "u")), str(tmp_path / "p1.jsonl"))
    p1 = read_examples(tmp_path / "p1.jsonl")
    j1 = fake_judge(
        p1,
        SOURCE_Q1,
        lambda _e: {Q_DOMAIN: domain, **_universal_answers(tax, {uni_name: bad_key})},
    )
    write_examples(tmp_path / "j1.jsonl", j1)
    advance(str(tmp_path / "j1.jsonl"), str(tmp_path / "p2.jsonl"), str(tmp_path / "r1.jsonl"))
    p2 = read_judged_examples(tmp_path / "p2.jsonl")
    j2 = fake_judge(p2, SOURCE_Q2, lambda _e: {Q_SUBDOMAIN: subdomain})
    write_examples(tmp_path / "j2.jsonl", j2)
    advance(str(tmp_path / "j2.jsonl"), str(tmp_path / "p3.jsonl"), str(tmp_path / "r2.jsonl"))
    p3 = read_judged_examples(tmp_path / "p3.jsonl")
    j3 = fake_judge(p3, SOURCE_Q3, lambda _e: {Q_INTENT: intent})
    write_examples(tmp_path / "j3.jsonl", j3)
    report = finalize(
        str(tmp_path / "j3.jsonl"),
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "fin.rejects.jsonl"),
        str(tmp_path / "report.json"),
    )
    assert report["kept"] == 1
    assert report["dropped_invalid_universal"] == 1
    assert uni_name not in report["labels"]
    ex = read_examples(tmp_path / "train.jsonl")[0]
    assert uni_name not in ex.labels["final"]
    assert report["labels"][Q_DOMAIN][domain] == 1


def _write_clean(tmp_path: Path, id_: str) -> Path:
    p = tmp_path / f"{id_}.jsonl"
    p.write_text(json.dumps({"id": id_, "state": "Test mesajı."}), encoding="utf-8")
    return p
