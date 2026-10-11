import pytest
from pydantic import ValidationError

from pusula.schema import Example, Question, answer_from_probs, argmax_key, probs_from_d3_answer, stable_id


def test_choice_validation():
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="x", criteria={"tek": "only one"})
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="x", criteria=["a", "b"])


def test_noul_and_score_validation():
    Question(type="noul", instructions="x")
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="x", criteria={"maybe": "?"})
    with pytest.raises(ValidationError):
        Question(type="score", instructions="x", criteria=["tek"])
    with pytest.raises(ValidationError):
        Question(type="score", instructions="x", criteria=["a", "a"])


def test_option_keys_and_texts(choice_q):
    assert choice_q.option_keys() == ["iade", "fatura", "teknik"]
    assert choice_q.option_texts() == ["İade ve değişim", "Fatura ve ödeme", "teknik"]
    noul = Question(type="noul", instructions="x", criteria={"true": "Evet, acil"})
    assert noul.option_keys() == ["false", "true"]
    assert noul.option_texts() == ["Hayır / yanlış", "Evet, acil"]
    score = Question(type="score", instructions="x", criteria=["düşük", "orta", "yüksek"])
    assert score.option_keys() == ["0", "1", "2"]
    assert score.option_texts()[2] == "Seviye 3/3: yüksek"


def test_answer_shapes_roundtrip(choice_q):
    a = answer_from_probs(choice_q, [0.2, 0.7, 0.1])
    assert a["choice"] == "fatura" and abs(sum(a["probabilities"].values()) - 1) < 1e-9
    assert probs_from_d3_answer(choice_q, a) == pytest.approx([0.2, 0.7, 0.1])

    noul = Question(type="noul", instructions="x")
    a = answer_from_probs(noul, [0.25, 0.75])
    assert a == {"type": "noul", "noul": 0.75}
    assert probs_from_d3_answer(noul, a) == pytest.approx([0.25, 0.75])

    score = Question(type="score", instructions="x", criteria=["a", "b", "c"])
    a = answer_from_probs(score, [0.0, 0.5, 0.5])
    assert a["score"] == pytest.approx(1.5) and a["legend"]["2"] == "c"
    assert argmax_key(score, probs_from_d3_answer(score, a)) in {"1", "2"}


def test_answer_rejects_bad_probs(choice_q):
    with pytest.raises(ValueError):
        answer_from_probs(choice_q, [0.5, 0.5])
    with pytest.raises(ValueError):
        probs_from_d3_answer(choice_q, {"error": "max_length_exceeded"})


def test_example_rejects_unknown_label(choice_q):
    with pytest.raises(ValidationError):
        Example(id="x", state="s", questions={"q": choice_q}, labels={"generator": {"q": "yok"}})
    with pytest.raises(ValidationError):
        Example(id="x", state="s", questions={"q": choice_q}, teacher={"q": [1.0]})


def test_stable_id_normalizes():
    assert stable_id("Merhaba  Dünya") == stable_id("merhaba dünya")
