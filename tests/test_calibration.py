import math

from pusula.calibrate import apply_bias
from pusula.inference import D3Engine
from pusula.schema import Question
from pusula.taxonomy import load_taxonomy


def test_apply_bias_shifts_and_normalizes():
    out = apply_bias([0.5, 0.5], [0.0, math.log(3)])
    assert abs(sum(out) - 1) < 1e-9
    assert abs(out[1] - 0.75) < 1e-9


def test_engine_calibrates_only_the_builtin_question():
    name, q = next((n, q) for n, q in load_taxonomy().universal.items() if q.type == "noul")
    calib = {"questions": {name: {"options": len(q.option_keys()), "bias": [0.0, math.log(3)]}}}
    engine = D3Engine(decider=None, calibration=calib)
    assert abs(engine._calibrate(name, q, [0.5, 0.5])[1] - 0.75) < 1e-9
    custom = Question(type="noul", instructions="Başka bir soru mu?")
    assert engine._calibrate(name, custom, [0.5, 0.5]) == [0.5, 0.5]
    assert engine._calibrate("baska", q, [0.5, 0.5]) == [0.5, 0.5]


def test_engine_ignores_mismatched_calibration():
    name, q = next(iter(load_taxonomy().universal.items()))
    calib = {"questions": {name: {"options": len(q.option_keys()) + 1, "bias": [0.0] * (len(q.option_keys()) + 1)}}}
    assert D3Engine(decider=None, calibration=calib)._calibration == {}


def test_batches_fit_the_budget_and_limit_padding():
    from pusula.inference import D3Engine

    eng = D3Engine(decider=None, token_budget=4096)
    lengths = [200, 210, 220, 250, 1000, 1100]
    ends, start = [], 0
    while start < len(lengths):
        start = eng._batch_end(lengths, start)
        ends.append(start)
    assert ends == [4, 6]  # the long prompts are not padded together with the short ones
    assert D3Engine(decider=None, token_budget=300)._batch_end([200, 210, 220], 0) == 1  # budget
    assert eng._batch_end([5000], 0) == 1  # a prompt longer than the budget still runs alone
