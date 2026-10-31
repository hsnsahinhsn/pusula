"""d3 prompt contract and training-item construction (no model weights needed)."""
import random

from pusula import d3model
from pusula.d3train import build_items, micro_batches
from pusula.schema import Question

CODES = [chr(ord("A") + i) for i in range(26)]


class ChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and enable_thinking is False
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(c) % 1000 for c in t] for t in texts]}


class FakeDecider:
    tokenizer = ChatTokenizer()
    codes = CODES

    def encode(self, prompts):
        return self.tokenizer(prompts)["input_ids"]


def test_option_texts_follow_d3_format():
    choice = Question(type="choice", instructions="?", criteria={"iade": "Ürün iadesi", "diger": None})
    assert d3model.option_texts(choice) == ["iade: Ürün iadesi", "diger"]
    noul = Question(type="noul", instructions="?", criteria={"true": "Evet, acil"})
    assert d3model.option_texts(noul) == ["No / false", "Evet, acil"]
    score = Question(type="score", instructions="?", criteria=["düşük", "yüksek"])
    assert d3model.option_texts(score) == ["düşük", "yüksek"]


def test_render_contains_codes_and_state():
    q = Question(type="choice", instructions="Hangi ekip?", criteria={"a": "Birinci", "b": "İkinci", "c": None})
    text = d3model.render(ChatTokenizer(), "Kartım çalındı", q, CODES, order=[2, 0, 1])
    assert d3model.SYSTEM_PROMPT in text
    assert "State:\nKartım çalındı" in text
    assert "A: c\nB: a: Birinci\nC: b: İkinci" in text
    assert text.endswith("Reply with only the code of the best option.<assistant>")


def test_shuffled_order_only_for_choice():
    rng = random.Random(0)
    assert d3model.shuffled_order(Question(type="noul", instructions="?"), rng) is None
    assert d3model.shuffled_order(Question(type="score", instructions="?", criteria=["a", "b"]), rng) is None
    order = d3model.shuffled_order(Question(type="choice", instructions="?", criteria={k: None for k in "abcd"}), rng)
    assert sorted(order) == [0, 1, 2, 3]


def test_build_items_tracks_label_through_shuffle(example):
    ex = example.model_copy(deep=True)
    ex.labels["final"] = dict(ex.labels["generator"])
    items = build_items([ex], FakeDecider(), "final", random.Random(3))
    assert len(items) == len(ex.questions)
    for it in items:
        q = ex.questions[it.name]
        assert it.count == len(q.option_keys())
        # the code shown at the label position must be followed by the label's own text in the prompt
        prompt = "".join(chr(c) for c in it.ids)
        label_text = d3model.option_texts(q)[q.option_keys().index(ex.labels["final"][it.name])]
        code = CODES[it.label]
        assert f"\n{code}: {label_text[:20]}" in prompt
    batches = micro_batches(items, token_budget=10**6, rng=random.Random(1))
    assert sum(len(b) for b in batches) == len(items)


def test_max_prompts_balances_question_names(example):
    rows = []
    for i in range(20):
        ex = example.model_copy(deep=True)
        ex.id = f"e{i}"
        ex.labels["final"] = dict(ex.labels["generator"])
        rows.append(ex)
    items = build_items(rows, FakeDecider(), "final", random.Random(0), max_prompts=45)
    assert len(items) == 45
    names = {it.name for it in items}
    assert names == set(rows[0].labels["final"])


def test_max_prompts_weights_focus_names(example):
    rows = []
    for i in range(40):
        ex = example.model_copy(deep=True)
        ex.id = f"e{i}"
        ex.labels["final"] = dict(ex.labels["generator"])
        rows.append(ex)
    names = sorted(rows[0].labels["final"])
    n = 2 * (len(names) + 1)  # focus name weight 2 -> 4 prompts, every other name 2
    items = build_items(rows, FakeDecider(), "final", random.Random(0), max_prompts=n, weights={"aciliyet": 2.0})
    counts = {name: sum(it.name == name for it in items) for name in names}
    assert len(items) == n and counts["aciliyet"] == 4
    assert all(c == 2 for name, c in counts.items() if name != "aciliyet")


def test_teacher_labels_every_question(example):
    torch = __import__("pytest").importorskip("torch")
    from pusula.teacher import label_chunk

    class LogitDecider(FakeDecider):
        def logits(self, sequences, counts):
            k = max(counts)
            out = torch.full((len(sequences), k), -1e4)
            for i, c in enumerate(counts):
                out[i, :c] = torch.arange(c, dtype=torch.float)  # always prefers the last option
            return out

    ex = example.model_copy(deep=True)
    label_chunk(LogitDecider(), [ex], token_budget=2000)
    assert set(ex.labels["d3"]) == set(ex.questions)
    for name, q in ex.questions.items():
        assert ex.labels["d3"][name] == q.option_keys()[-1]
        assert abs(sum(ex.teacher[name]) - 1) < 1e-5


def test_smoothed_ce_ignores_padded_codes():
    torch = __import__("pytest").importorskip("torch")
    from pusula.d3train import smoothed_ce

    logits = torch.tensor([[2.0, 0.0, -1e4, -1e4], [0.0, 1.0, 0.5, 3.0]])
    labels = torch.tensor([0, 3])
    loss = smoothed_ce(logits, labels, [2, 4], 0.1)
    ref = (torch.nn.functional.cross_entropy(logits[:1, :2], labels[:1], label_smoothing=0.1)
           + torch.nn.functional.cross_entropy(logits[1:], labels[1:], label_smoothing=0.1)) / 2
    assert torch.allclose(loss, ref, atol=1e-5) and loss.item() < 2


def test_distill_kl_uses_teacher_rows_only():
    torch = __import__("pytest").importorskip("torch")
    from pusula.d3train import Item, distill_kl

    logits = torch.tensor([[2.0, 0.0, -1e4], [0.0, 1.0, 0.5]])
    batch = [Item("a", "choice", [1], 0, 2, [0.9, 0.1]), Item("b", "choice", [1], 1, 3, None)]
    kl = distill_kl(logits, batch)
    p = torch.softmax(logits[0, :2], -1)
    ref = (torch.tensor([0.9, 0.1]) * (torch.tensor([0.9, 0.1]).log() - p.log())).sum()
    assert torch.allclose(kl, ref, atol=1e-5)
    assert distill_kl(logits, [batch[1]]).item() == 0.0


def test_wide_intent_question(example):
    import random

    from pusula.distill import WIDE_QUESTION, add_wide_intent

    ex = example.model_copy(deep=True)
    ex.labels["final"] = dict(ex.labels["generator"])
    assert add_wide_intent(ex, random.Random(1))
    q = ex.questions[WIDE_QUESTION]
    assert 30 <= len(q.option_keys()) <= 60 and ex.labels["final"]["niyet"] in q.option_keys()
    assert ex.labels["final"][WIDE_QUESTION] == ex.labels["final"]["niyet"]


def test_teacher_labels_only_selected_questions(example):
    torch = __import__("pytest").importorskip("torch")
    from pusula.teacher import label_chunk

    class LogitDecider(FakeDecider):
        def logits(self, sequences, counts):
            out = torch.full((len(sequences), max(counts)), -1e4)
            for i, c in enumerate(counts):
                out[i, :c] = torch.arange(c, dtype=torch.float)
            return out

    ex = example.model_copy(deep=True)
    first = next(iter(ex.questions))
    label_chunk(LogitDecider(), [ex], token_budget=2000, source="pusula", questions={first})
    assert set(ex.labels["pusula"]) == {first} and set(ex.teacher) == {first}


def _noul_rows(n_pos, n_neg):
    from pusula.schema import Example

    q = Question(type="noul", instructions="Hakaret var mı?", criteria={"true": "Evet"})
    return [Example(id=f"r{i}", state=f"mesaj {i}", questions={"hakaret": q},
                    labels={"final": {"hakaret": "true" if i < n_pos else "false"}}) for i in range(n_pos + n_neg)]


def _with_filler(rows):
    filler = _noul_rows(0, len(rows))
    for ex in filler:
        ex.id = "f" + ex.id
        ex.questions = {"diger": ex.questions["hakaret"]}
        ex.labels = {"final": {"diger": "false"}}
    return rows + filler


def _rare(items):
    return sum(it.name == "hakaret" and it.label == 1 for it in items)


def test_class_balance_boost_is_capped_at_3x():
    # 30 positives in 200 rows, a 20-row share: natural 3, at most 3x = 9 (before the cap: all 10 of the even split)
    rows = _with_filler(_noul_rows(30, 170))
    items = build_items(rows, FakeDecider(), "final", random.Random(0), max_prompts=40, class_balance=True)
    assert len(items) == 40 and _rare(items) == 9
    # 5 positives in 200: natural 0.5, cap ceil(1.5) = 2
    rows = _with_filler(_noul_rows(5, 195))
    assert _rare(build_items(rows, FakeDecider(), "final", random.Random(0), max_prompts=40, class_balance=True)) == 2


def test_class_balance_only_in_named_questions():
    # same 30-in-200 positives in two names: balanced only where named (9 of 20), natural elsewhere
    rows = _noul_rows(30, 170)
    other = _noul_rows(30, 170)
    for ex in other:
        ex.id = "o" + ex.id
        ex.questions = {"diger": ex.questions["hakaret"]}
        ex.labels = {"final": {"diger": ex.labels["final"]["hakaret"]}}
    items = build_items(rows + other, FakeDecider(), "final", random.Random(0), max_prompts=40,
                        class_balance=frozenset({"hakaret"}))
    assert _rare(items) == 9
    assert sum(it.name == "diger" and it.label == 1 for it in items) < 9


def test_balance_flag_parsing():
    from types import SimpleNamespace

    from pusula.d3train import _balance
    assert _balance(SimpleNamespace(balance_only="", no_class_balance=False)) is True
    assert _balance(SimpleNamespace(balance_only="", no_class_balance=True)) is False
    assert _balance(SimpleNamespace(balance_only="eksik_bilgi,hakaret", no_class_balance=True)) == {"eksik_bilgi", "hakaret"}


def test_class_balance_keeps_the_type_mix_of_own_questions():
    from pusula.schema import Example

    # the writer's own question s1: unique choice keys mixed with noul and score (57/30/13 %), as in the factory
    rng, rows = random.Random(3), []
    for i in range(1000):
        r = rng.random()
        if r < 0.57:
            q = Question(type="choice", instructions="?", criteria={f"k{i}a": None, f"k{i}b": None})
            lab = f"k{i}a"
        elif r < 0.87:
            q, lab = Question(type="noul", instructions="?", criteria={"true": "Evet"}), rng.choice(["true", "false"])
        else:
            q, lab = Question(type="score", instructions="?", criteria=["az", "orta", "çok"]), str(rng.randrange(3))
        rows.append(Example(id=f"s{i}", state="x", questions={"s1": q}, labels={"final": {"s1": lab}}))
    filler = [Example(id=f"f{i}", state="x", questions={"diger": Question(type="noul", instructions="?", criteria={"true": "E"})},
                      labels={"final": {"diger": "false"}}) for i in range(1000)]
    items = build_items(rows + filler, FakeDecider(), "final", random.Random(0), max_prompts=600, class_balance=True)
    own = [it.qtype for it in items if it.name == "s1"]
    assert len(own) == 300 and 0.30 <= sum(t != "choice" for t in own) / len(own) <= 0.56  # before: almost all choice


def test_rare_first_ties_are_random():
    from pusula.d3train import rare_first
    from pusula.schema import Example

    q = Question(type="choice", instructions="?", criteria={k: None for k in "abc"})
    rows = [(Example(id=f"t{i}", state="x", questions={"tur": q}, labels={"final": {"tur": "abc"[i % 3]}}), "tur")
            for i in range(30)]
    smallest = set()
    for seed in range(20):  # share 5 over three equal values: one value gets 1 row, the other two get 2
        got = [ex.labels["final"]["tur"] for ex, _ in rare_first(rows, "final", random.Random(seed), 5)[:5]]
        assert len(got) == 5 and sorted(got.count(k) for k in "abc") == [1, 2, 2]
        smallest.add(min("abc", key=got.count))
    assert len(smallest) > 1  # before: alphabetical, the same value every time


def test_build_items_empty_and_val_set_shuffles():
    from pusula.d3train import val_set

    assert build_items([], FakeDecider(), "final", None) == []
    q = Question(type="choice", instructions="?", criteria={k: None for k in "abcdefgh"})
    from pusula.schema import Example

    rows = [Example(id=f"v{i}", state="x", questions={"tur": q}, labels={"final": {"tur": "a"}}) for i in range(40)]
    items = val_set(rows, FakeDecider(), 17, 0)
    assert {it.label for it in items} != {0}  # the first option is not always the answer
    assert [it.label for it in items] == [it.label for it in val_set(rows, FakeDecider(), 17, 0)]  # fixed seed


def test_check_gate_fails_on_empty_val(tmp_path, monkeypatch):
    import pytest

    from pusula import d3train

    monkeypatch.setattr(d3train, "TokenizerDecider", lambda _dir: FakeDecider())
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("".join(ex.model_dump_json() + "\n" for ex in _noul_rows(3, 10)))
    (data / "val.jsonl").write_text("")
    with pytest.raises(SystemExit):
        d3train.main(["check", "--d3", "x", "--data", str(data)])
    (data / "val.jsonl").write_text("".join(ex.model_dump_json() + "\n" for ex in _noul_rows(1, 2)))
    d3train.main(["check", "--d3", "x", "--data", str(data), "--max-prompts", "5"])


def test_save_due_on_steps_and_on_the_clock():
    from pusula.d3train import save_due

    assert save_due(100, 440, 100, 30, 5) is True and save_due(440, 440, 100, 30, 5) is True  # schedule, last step
    assert save_due(37, 440, 100, 30, 29 * 60) is False
    assert save_due(37, 440, 100, 30, 30 * 60) == "clock"  # ranks agree on it before saving
    assert save_due(37, 440, 100, 0, 10 ** 6) is False  # off by default


def test_checkpoint_swap_and_recover(tmp_path):
    """A save is written to <name>.tmp and swapped in; a kill between the two renames leaves <name>.old, which is used."""
    from pusula.d3train import _fresh, _recover, _swap_in

    dst = tmp_path / "last"
    for step in (1, 2):
        tmp = _fresh(tmp_path / "last.tmp")
        tmp.mkdir()
        (tmp / "trainer_state.json").write_text(str(step))
        _swap_in(tmp, dst)
    assert (dst / "trainer_state.json").read_text() == "2"
    assert not (tmp_path / "last.old").exists() and not (tmp_path / "last.tmp").exists()
    dst.rename(tmp_path / "last.old")  # killed after the first rename of the next save
    (tmp_path / "last.tmp").mkdir()
    (tmp_path / "last.tmp" / "half").write_text("x")
    _recover(dst)
    assert (dst / "trainer_state.json").read_text() == "2"
    assert not (tmp_path / "last.old").exists() and not (tmp_path / "last.tmp").exists()


def _ddp_worker(rank, world, port, out):
    import torch.distributed as dist

    from pusula.d3train import ddp_probe

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    ok, gbps = ddp_probe(dist, "cpu", rank, world, mb=1, small=3)
    out[rank] = (ok, gbps > 0)
    dist.destroy_process_group()


def test_ddp_probe_two_processes():
    """The pre-training link test passes on a healthy 2-process group (gloo on CPU stands in for NCCL)."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_ddp_worker, args=(2, port, out), nprocs=2)
    assert dict(out) == {0: (True, True), 1: (True, True)}


def test_ddp_probe_catches_corrupt_sums():
    """A link that returns wrong sums (not only one that hangs) fails the test on every rank."""
    from types import SimpleNamespace

    from pusula.d3train import ddp_probe

    class Bad:
        ReduceOp = SimpleNamespace(MAX="max")
        calls = 0

        def all_reduce(self, x, op=None):
            if op is None:
                Bad.calls += 1
                x.fill_(3.0 if Bad.calls != 5 else 2.0)  # world 2: 1 + 2 = 3; one small all-reduce comes back wrong

    ok, _ = ddp_probe(Bad(), "cpu", 0, 2, mb=1, small=4)
    assert not ok


def test_eval_loads_the_model_once_for_many_sets(tmp_path, monkeypatch):
    import json

    import pytest
    import torch

    from pusula import d3train

    loads = []

    class EvalDecider(FakeDecider):
        backbone = torch.nn.Identity()

        def logits(self, ids, counts):
            return torch.tensor([[2.0] + [0.0] * (max(counts) - 1) for _ in ids])

    def load(*args, **kwargs):
        loads.append(args)
        return EvalDecider()

    monkeypatch.setattr(d3model.D3Decider, "load", staticmethod(load))
    files, outs = [], []
    for k in range(3):
        f = tmp_path / f"eval_{k}.jsonl"
        f.write_text("".join(ex.model_dump_json() + "\n" for ex in _noul_rows(1, 2)))
        files.append(str(f))
        outs.append(str(tmp_path / "r" / f"v_{k}.json"))
    d3train.main(["eval", "--d3", "x", "--precision", "bf16", "--data", *files, "--out", *outs])
    assert len(loads) == 1
    for f, o in zip(files, outs):
        rep = json.loads(open(o).read())
        assert rep["data"] == f and rep["n"] == 3 and rep["model"] == "d3-base"
    assert not list((tmp_path / "r").glob("*.tmp"))
    with pytest.raises(SystemExit):
        d3train.main(["eval", "--d3", "x", "--data", *files, "--out", outs[0]])
