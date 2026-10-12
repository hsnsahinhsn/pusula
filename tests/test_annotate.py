from pusula.annotate import (
    Ledger,
    agreement,
    cost,
    disputed,
    parse_json,
    system_prompt,
    user_prompt,
    validate,
)
from pusula.schema import Example, Question
from pusula.taxonomy import load_taxonomy


def _example(extra=None):
    tax = load_taxonomy()
    p = next(tax.intent_paths())
    qs = tax.questions_for(p)
    if extra:
        qs["ozel"] = extra
    return Example(id="x1", state="Kartımdan bilgim dışında para çekildi", questions=qs), p


def test_cost_counts_cache_and_batch():
    u = {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0,
         "output_tokens": 1_000_000}
    assert cost("claude-opus-5-5", u) == 4.0 + 0.2 + 20.0
    assert cost("claude-opus-5-5", u, batch=True) == (4.0 + 0.2 + 20.0) / 2
    assert cost("claude-sonnet-5-5", {"cache_creation_input_tokens": 1_000_000}, ttl="1h") == 4.0


def test_system_prompt_lists_taxonomy_and_policy():
    tax = load_taxonomy()
    s = system_prompt(tax)
    assert all(d in s for d in tax.domains)
    assert "insana_devret" in s and "Yakın" in s and "etiketleme politikası" in s.lower()


def test_user_prompt_shows_extras_but_never_labels():
    ex, _ = _example(Question(type="noul", instructions="İade istiyor mu?"))
    u = user_prompt(ex)
    assert ex.state in u and "ozel" in u and "İade istiyor mu?" in u


def test_parse_and_validate():
    ex, p = _example(Question(type="choice", instructions="Hangi ekip?", criteria={"kart": "Kart", "fraud": "Fraud"}))
    out = parse_json('düşündüm...\n{"alan": "%s", "alt_alan": "%s", "niyet": "%s", "aciliyet": 3, "eylem_gerekli": true,'
                     ' "insana_devret": "true", "guvenlik": "guvenli", "duygu": "endiseli", "kisisel_veri": false,'
                     ' "ozel": "fraud", "confidence": "high", "drop": false, "note": ""}' % (p.domain, p.subdomain, p.intent))
    labels, errors = validate(load_taxonomy(), ex, out)
    assert errors == []
    assert labels["aciliyet"] == "3" and labels["eylem_gerekli"] == "true" and labels["kisisel_veri"] == "false"
    assert labels["ozel"] == "fraud"
    out["niyet"] = "yok_boyle_bir_niyet"
    out["ozel"] = "baska"
    labels, errors = validate(load_taxonomy(), ex, out)
    assert "niyet" not in labels and "ozel" not in labels and len(errors) == 2
    assert parse_json("json yok") is None


def test_disputed_routing_is_one_decision():
    a = {"alan": "x", "alt_alan": "y", "niyet": "z", "aciliyet": "1"}
    assert disputed(a, dict(a)) == []
    assert disputed(a, {**a, "niyet": "w"}) == ["alan", "alt_alan", "niyet"]
    assert disputed(a, {**a, "aciliyet": "2"}) == ["aciliyet"]


def test_agreement_and_ledger(tmp_path):
    ra = {"1": {"labels": {"q": "a"}}, "2": {"labels": {"q": "b"}}}
    rb = {"1": {"labels": {"q": "a"}}, "2": {"labels": {"q": "a"}}, "3": {"labels": {"q": "a"}}}
    assert agreement(ra, rb) == {"q": (1, 2)}
    led = Ledger(tmp_path / "spend.jsonl", cap=1.0)
    led.add({"cost": 0.6})
    assert led.room() and not led.room(reserve=0.5)
    assert Ledger(tmp_path / "spend.jsonl", cap=1.0).total == 0.6


def test_check_questions_sizes():
    from pusula.annotate import check_questions

    def q(tip, n):
        return {"talimat": "Soru?", "tip": tip, "secenekler": [{"anahtar": f"s{i}", "aciklama": "x"} for i in range(n)]}

    ex, _ = _example()
    good = {"sorular": [q("noul", 0), q("choice", 4), q("choice", 12)]}
    labels, errors = check_questions(ex, good)
    assert errors == [] and set(labels) == {"adhoc_1", "adhoc_2", "adhoc_3"}
    labels, errors = check_questions(ex, {"sorular": [q("noul", 0), q("choice", 4), q("choice", 5)]})
    assert labels == {} and errors


def test_validate_extras_and_specs():
    from pusula.annotate import _jaccard, _shingles, message_specs, validate_extras

    ex, _ = _example(Question(type="noul", instructions="İade istiyor mu?"))
    assert validate_extras(ex, {"ozel": True}) == ({"ozel": "true"}, [])
    assert validate_extras(ex, {})[1] == ["ozel=None"]
    s1, s2 = message_specs(50, seed=3), message_specs(50, seed=3)
    assert s1 == s2 and len({s["path"] for s in s1}) > 20
    assert _jaccard(_shingles("kargom gelmedi hala"), _shingles("kargom gelmedi hala")) == 1.0
