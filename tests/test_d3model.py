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
