import pytest

torch = pytest.importorskip("torch")

from pusula import curate  # noqa: E402


def _unit(n, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_dedup_drops_near_duplicates(example):
    emb = _unit(4)
    emb[2] = emb[0]  # exact duplicate of the first
    keep = curate.dedup([example] * 4, emb, threshold=0.95)
    assert keep == [0, 1, 3]


def test_hard_question_mixes_other_subdomains(monkeypatch, example, tax):
    n_intents = len(list(tax.intent_paths()))
    monkeypatch.setattr(curate, "embed", lambda texts, **k: _unit(len(texts), seed=len(texts)))
    ex = example.model_copy(deep=True)
    ex.labels["final"] = dict(ex.labels["generator"])
    added = curate.add_hard_question([ex], _unit(1, seed=99), k=5)
    assert added == 1 and n_intents > 6
    q = ex.questions[curate.HARD_QUESTION]
    true = ex.labels["final"]["niyet"]
    assert true in q.criteria and len(q.criteria) == 6
    own_sub = tax.domains[ex.labels["final"]["alan"]].subdomains[ex.labels["final"]["alt_alan"]].intents
    assert [k for k in q.criteria if k in own_sub] == [true]
    assert ex.labels["final"][curate.HARD_QUESTION] == true
