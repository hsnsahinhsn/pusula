"""Student model, inference engine and training helpers on a tiny random ModernBERT (CPU, no downloads)."""
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from pusula import train as T  # noqa: E402
from pusula.model import PusulaModel  # noqa: E402


def test_decide_returns_d3_shapes(tiny_engine, choice_q):
    out = tiny_engine.decide("Faturamda fazladan ücret var.", {
        "ekip": choice_q,
        "acil": {"type": "noul", "instructions": "Acil mi?"},
        "risk": {"type": "score", "instructions": "Risk?", "criteria": ["düşük", "orta", "yüksek"]},
    })
    assert out["ekip"]["choice"] in choice_q.option_keys()
    assert sum(out["ekip"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= out["acil"]["noul"] <= 1.0
    assert 0.0 <= out["risk"]["score"] <= 2.0


def test_decide_many_matches_decide(tiny_engine, choice_q):
    states = ["kısa mesaj", "çok daha uzun bir mesaj " * 10]
    many = tiny_engine.decide_many([(s, {"q": choice_q}) for s in states])
    for s, m in zip(states, many):
        single = tiny_engine.decide(s, {"q": choice_q})
        for k, p in single["q"]["probabilities"].items():
            assert m["q"]["probabilities"][k] == pytest.approx(p, abs=1e-4)


def test_route_follows_taxonomy(tiny_engine, tax):
    out = tiny_engine.route("İnternetim iki gündür kesik.", tax)
    d, s, i = out["path"].split(".")
    assert i in tax.domains[d].subdomains[s].intents
    assert "aciliyet" in out["answers"]


def test_temperature_changes_confidence(tiny_engine, choice_q):
    base = tiny_engine.decide("x", {"q": choice_q})["q"]["confidence"]
    tiny_engine.model.temperature["choice"] = 50.0
    flat = tiny_engine.decide("x", {"q": choice_q})["q"]["confidence"]
    tiny_engine.model.temperature["choice"] = 1.0
    assert flat <= base + 1e-6 and flat == pytest.approx(1 / 3, abs=0.05)


def test_save_load_roundtrip(tmp_path, tiny_engine, monkeypatch, tokenizer):
    import transformers

    model = tiny_engine.model
    model.temperature = {"choice": 1.5, "noul": 0.8, "score": 1.0}
    model.save(tmp_path / "m", max_length=128)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    loaded, _, cfg = PusulaModel.load(tmp_path / "m", attn_implementation="sdpa")
    assert cfg["max_length"] == 128 and loaded.temperature["choice"] == 1.5
    assert torch.allclose(loaded.scorer.weight, model.scorer.weight)


def test_training_step_and_calibration(tiny_engine, example, tokenizer):
    enc = tiny_engine.enc
    ex = example.model_copy(deep=True)
    ex.labels["final"] = dict(ex.labels["generator"])
    q = ex.questions["alan"]
    ex.teacher["alan"] = [1.0 / len(q.option_keys())] * len(q.option_keys())
    data = T.Encoded([ex, ex], enc)
    assert len(data.items) == 2 * len(ex.questions)
    batches = T.make_batches(data, enc, token_budget=4096, max_options=4, rng=random.Random(0))
    assert sum(len(b) for b in batches) == len(data.items)
    model = tiny_engine.model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for batch in batches:
        logits, _ = T.forward_batch(model, enc, data, batch, torch.device("cpu"))
        assert logits.shape[1] <= 4
        loss = T.batch_loss(logits, batch, alpha=0.5, device=torch.device("cpu"))
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
    preds = T.predict(model, enc, data, torch.device("cpu"), token_budget=4096)
    assert len(preds) == len(data.items)
    m = T.metrics(preds)
    assert 0.0 <= m["macro_acc"] <= 1.0 and 0.0 <= m["ece"] <= 1.0
    temps = T.fit_temperatures(preds)
    assert all(t > 0 for t in temps.values())


def test_subsample_keeps_label():
    item = T.Item(state=0, name="q", qtype="choice", options=[[i] for i in range(20)], label=13,
                  soft=[0.05] * 20)
    idx, lab, soft = T.subsample(item, 8, random.Random(1))
    assert len(idx) == 8 and idx[lab] == 13 and sum(soft) == pytest.approx(1.0)
