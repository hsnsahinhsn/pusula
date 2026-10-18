import json

import pytest

from pusula.taxonomy import KEY_RE, MAX_SIBLINGS, ROUTING_QUESTIONS


def test_structure(tax):
    assert 15 <= len(tax.domains) <= MAX_SIBLINGS
    paths = list(tax.intent_paths())
    assert len(paths) == len({str(p) for p in paths})
    for d in tax.domains.values():
        assert KEY_RE.match(d.key) and d.name and d.description
        for s in d.subdomains.values():
            assert 2 <= len(s.intents) <= MAX_SIBLINGS
            assert all(desc.strip() for desc in s.intents.values())


def test_universal_questions(tax):
    assert {"aciliyet", "eylem_gerekli", "insana_devret", "guvenlik", "duygu", "kisisel_veri"} <= set(tax.universal)
    assert tax.universal["aciliyet"].type == "score"
    assert "guvenli" in tax.universal["guvenlik"].option_keys()


def test_questions_for_path(tax):
    path = list(tax.intent_paths())[42]
    qs = tax.questions_for(path)
    assert set(ROUTING_QUESTIONS) <= set(qs)
    for name, label in path.as_labels().items():
        assert label in qs[name].option_keys()
    assert tax.describe(path).count(">") == 2


def test_to_dict_is_json(tax):
    data = json.loads(json.dumps(tax.to_dict(), ensure_ascii=False))
    assert set(data["domains"]) == set(tax.domains)


def test_agent_questions(tax):
    assert set(tax.agent) == {"onay_gerekli", "islem_riski", "netlestirme_gerekli", "veri_disari", "gorev_tamamlandi",
                              "model_secimi"}
    assert tax.agent["islem_riski"].type == "score" and len(tax.agent["islem_riski"].option_keys()) == 4
    assert tax.agent["gorev_tamamlandi"].option_keys() == ["tamamlandi", "kismen", "tamamlanmadi"]
    assert "agent" in tax.to_dict() and not set(tax.agent) & set(tax.universal)


def test_topic_questions():
    from pusula.taxonomy import topic_questions

    q, intents = topic_questions({"a": "Birinci", "b": {"description": "İkinci", "intents": {"x": "X", "y": "Y"}}})
    assert q.option_keys() == ["a", "b"] and list(intents) == ["b"]
    assert intents["b"].option_keys() == ["x", "y"]
    for bad in ({"tek": "yalnız bir konu"}, {"a": "A", "b": {"description": "B", "intents": {"x": "tek niyet"}}},
                {"a": "A", "b": {"aciklama": "yanlış anahtar"}}, ["a", "b"]):
        with pytest.raises(ValueError):
            topic_questions(bad)
