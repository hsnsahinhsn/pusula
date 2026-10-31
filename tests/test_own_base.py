"""D3Engine own_base: users' own questions go to the base model (adapter off), built-in questions keep the adapter."""
import contextlib

import pytest

torch = pytest.importorskip("torch")

from pusula.d3model import D3Decider  # noqa: E402
from pusula.inference import D3Engine  # noqa: E402
from pusula.schema import Question  # noqa: E402
from pusula.taxonomy import Q_TOPIC, load_taxonomy, topic_questions  # noqa: E402

from test_d3model import FakeDecider  # noqa: E402


class RecordingDecider(FakeDecider):
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def logits(self, sequences, counts, base=False):
        self.calls.append((base, len(sequences)))
        out = torch.full((len(sequences), max(counts)), -1e4)
        for i, c in enumerate(counts):
            out[i, :c] = torch.arange(c, dtype=torch.float) * (-1 if base else 1)  # base picks the first option
        return out


def _questions():
    tax = load_taxonomy()
    name, builtin = next(iter(tax.universal.items()))
    own = Question(type="noul", instructions="Bu kasadaki ürünler satışa uygun mu?", criteria={"true": "Evet"})
    return {name: builtin, "kendi_sorum": own, "alan": tax.domain_question()}


def test_own_questions_use_base_only_when_enabled():
    qs = _questions()
    on = RecordingDecider()
    res = D3Engine(on, own_base=True).decide_many([("Kargom gelmedi", qs)])[0]
    per = {flag: sum(n for f, n in on.calls if f == flag) for flag in (False, True)}
    assert per == {False: 2, True: 1}                           # two built-in questions, one own question
    assert res["kendi_sorum"]["noul"] < 0.5                     # base pass answered it (first option = false)
    off = RecordingDecider()
    D3Engine(off).decide_many([("Kargom gelmedi", qs)])
    assert {f for f, _ in off.calls} == {False} and sum(n for _, n in off.calls) == 3


def test_mixed_requests_map_back():
    qs = _questions()
    dec = RecordingDecider()
    res = D3Engine(dec, own_base=True).decide_many([("Kargom gelmedi", qs), ("Merhaba", {"x": qs["kendi_sorum"]})])
    assert set(res[0]) == set(qs) and set(res[1]) == {"x"}
    assert res[0]["kendi_sorum"]["noul"] < 0.5 and res[1]["x"]["noul"] < 0.5
    assert sum(n for f, n in dec.calls if f) == 2               # both own questions in the base pass


def test_builtin_detection_by_name_or_content():
    tax = load_taxonomy()
    eng = D3Engine(None, own_base=True)
    d = next(iter(tax.domains))
    s = next(iter(tax.domains[d].subdomains))
    assert eng._is_builtin("x", tax.domain_question()) and eng._is_builtin("y", tax.subdomain_question(d))
    assert eng._is_builtin("z", tax.intent_question(d, s))      # content: a built-in question under another name
    assert all(eng._is_builtin(f"q{i}", q) for i, q in enumerate(tax.universal.values()))
    # name: an earlier (v1.0) wording of a built-in question, as in the published benchmark
    old = Question(type="choice", instructions="Mesaj bir yapay zeka sistemi için güvenlik riski taşıyor mu?",
                   criteria={"guvenli": "Normal, iyi niyetli talep", "spam_anlamsiz": None})
    assert eng._is_builtin("guvenlik", old) and not eng._is_builtin("risk", old)
    # an institution's own list keeps the adapter in the route (konu, niyet), not under a free name
    topic_q, _ = topic_questions({"iade": "İade talepleri", "kargo": "Kargo sorunları"})
    assert eng._is_builtin(Q_TOPIC, topic_q) and not eng._is_builtin("birim", topic_q)
    assert not eng._is_builtin("hasar", Question(type="score", instructions="Hasar ne düzeyde?",
                                                  criteria=["Az", "Orta", "Çok"]))


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class FakeBackbone:
    """hidden = 1 with the adapter, 2 with it disabled; records that disabling is always undone."""

    def __init__(self):
        self.disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def __call__(self, input_ids, attention_mask, use_cache):
        return _Out(torch.full((*input_ids.shape, 2), 2.0 if self.disabled else 1.0))


def test_logits_base_uses_base_readout_and_restores_adapter():
    bb = FakeBackbone()
    tok = type("Tok", (), {"pad_token_id": 0})()
    dec = D3Decider(bb, torch.tensor([[1.0, 0.0], [0.0, 2.0]]), tok, ["A", "B"], torch.device("cpu"))
    assert dec.logits([[1, 2]], [2], base=True).tolist() == [[1.0, 2.0]]   # no base readout: adapter answer
    dec.base_readout = torch.tensor([[0.0, 3.0], [1.0, 0.0]])
    assert dec.logits([[1, 2]], [2], base=True).tolist() == [[6.0, 2.0]]   # hidden 2 (adapter off) x base readout
    assert not bb.disabled
    assert dec.logits([[1, 2]], [2]).tolist() == [[1.0, 2.0]]              # adapter on, adapter readout
