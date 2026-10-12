from pusula.jsonl import read_examples, write_examples
from pusula.judge import answer_schema, build_prompt, lettered, majority, refreshed, system_prompt
from pusula.relabel import apply, score
from pusula.schema import Example, Question
from pusula.taxonomy import load_taxonomy

OLD = Question(type="noul", instructions="Eski tanım: insana devredilmeli mi?")


def _ex(i="a", **labels):
    tax = load_taxonomy()
    qs = {"insana_devret": OLD, "duygu": tax.universal["duygu"]}
    return Example(id=i, state="Kazada panikledim, ne yapmalıyım?", questions=qs,
                   labels=labels or {"final": {"insana_devret": "true", "duygu": "endiseli"}},
                   teacher={"insana_devret": [0.2, 0.8]})


def test_policy_prompt_and_refresh():
    assert "etiketleme politikası" in system_prompt(True).lower() and system_prompt(False) != system_prompt(True)
    ex = refreshed(_ex(), {"insana_devret"})
    assert ex.questions["insana_devret"] == load_taxonomy().universal["insana_devret"]
    assert ex.labels == {} and ex.teacher == {}


def test_rationale_schema_and_majority():
    ex = _ex()
    order = lettered(ex)
    schema = answer_schema(order, rationale=True)
    assert list(schema["properties"])[0] == "gerekce" and "gerekce" in schema["required"]
    assert "gerekce" in build_prompt(ex, order, rationale=True)
    assert majority([{"q": "a"}, {"q": "b"}, {"q": "b"}]) == {"q": "b"}
    assert majority([{"q": "a"}, {"q": "b"}]) == {"q": "a"}


def test_apply_swaps_definitions_and_labels(tmp_path):
    write_examples(tmp_path / "inp.jsonl", [_ex("a"), _ex("b")])
    pred = [_ex("a", final={"insana_devret": "true"}, relabel={"insana_devret": "false"})]
    write_examples(tmp_path / "pred.jsonl", pred)
    apply(str(tmp_path / "inp.jsonl"), str(tmp_path / "pred.jsonl"), "relabel", ["insana_devret"],
          str(tmp_path / "out.jsonl"))
    out = {ex.id: ex for ex in read_examples(tmp_path / "out.jsonl")}
    a, b = out["a"], out["b"]
    assert a.questions["insana_devret"] == load_taxonomy().universal["insana_devret"]
    assert a.labels["final"] == {"insana_devret": "false", "duygu": "endiseli"} and a.teacher == {}
    assert "insana_devret" not in b.questions and b.labels["final"] == {"duygu": "endiseli"}


def test_score(tmp_path):
    write_examples(tmp_path / "p.jsonl", [_ex("a", gold={"insana_devret": "false"}, relabel={"insana_devret": "false"}),
                                          _ex("b", gold={"insana_devret": "false"}, relabel={"insana_devret": "true"})])
    s = score(str(tmp_path / "p.jsonl"), "relabel")
    assert s["insana_devret"]["acc"] == 0.5 and s["insana_devret"]["confusion"] == {"false->false": 1, "false->true": 1}
