"""Judge prompt layouts: the default stays as before; questions-first + fixed order gives a shared prompt prefix."""
import random

from pusula.judge import arranged, build_prompt, code, decode, lettered, shared_names
from pusula.schema import Example, Question
from pusula.taxonomy import load_taxonomy


def _example(ex_id: str, state: str, own: bool = True) -> Example:
    tax = load_taxonomy()
    qs = {"s1": Question(type="noul", instructions="Kişi iade istiyor mu?")} if own else {}
    qs.update({"alan": tax.domain_question(), **tax.universal})  # own question first on purpose
    return Example(id=ex_id, state=state, questions=qs)


def _old_prompt(ex: Example, seed: int = 0) -> str:
    """The judge prompt before the layout option (19 Oct 2026), for the back-compat check."""
    order = {}
    for name, q in ex.questions.items():
        keys = q.option_keys()
        if q.type != "score":
            keys = keys[:]
            random.Random(f"{seed}:{ex.id}:{name}").shuffle(keys)
        order[name] = keys
    lines = ["Durum:", ex.state, ""]
    for n, (name, q) in enumerate(ex.questions.items(), 1):
        texts = dict(zip(q.option_keys(), q.option_texts()))
        lines.append(f"Soru {n} ({name}): {q.instructions}")
        lines += [f"  {code(i)}) {texts[k]}" for i, k in enumerate(order[name])]
        lines.append("")
    lines.append("Her soru için yalnızca seçtiğin harfi ver.")
    return "\n".join(lines)


def test_default_layout_is_unchanged():
    ex = _example("a", "Siparişim gelmedi, paramı geri istiyorum.")
    view, lead = arranged(ex)
    assert lead == 0 and list(view.questions) == list(ex.questions)
    assert build_prompt(view, lettered(view, 0)) == _old_prompt(ex)


def test_questions_first_shares_the_prefix():
    a, b = _example("a", "Siparişim gelmedi."), _example("b", "Faturam yanlış geldi, düzeltin lütfen.", own=False)
    fixed = frozenset(shared_names())
    prompts = []
    for ex in (a, b):
        view, lead = arranged(ex, "questions-first")
        assert lead == 15 and list(view.questions)[:lead] == [n for n in shared_names() if n in ex.questions]
        prompts.append(build_prompt(view, lettered(view, 0, fixed), lead=lead))
    head = prompts[0].split("Durum:")[0]
    assert head == prompts[1].split("Durum:")[0]  # identical up to the state
    assert "Soru 15 (" in head and "(s1)" not in head
    assert prompts[0].index("Durum:") < prompts[0].index("(s1)")  # own questions follow the state
    assert prompts[1].rstrip().endswith("yalnızca seçtiğin harfi ver.") and "Faturam yanlış" in prompts[1]


def test_fixed_order_only_for_builtin_questions():
    a, b = _example("a", "x"), _example("b", "y")
    fixed = frozenset(shared_names())
    oa, ob = lettered(a, 0, fixed), lettered(b, 0, fixed)
    assert all(oa[n] == ob[n] for n in fixed if n in oa)
    assert lettered(a, 0)["guvenlik"] != lettered(b, 0)["guvenlik"] or lettered(a, 0)["alan"] != lettered(b, 0)["alan"]


def test_decode_round_trip_with_fixed_order():
    view, lead = arranged(_example("a", "x"), "questions-first")
    order = lettered(view, 3, frozenset(shared_names()))
    want = {name: keys[-1] for name, keys in order.items()}
    letters = {name: code(len(keys) - 1) for name, keys in order.items()}
    assert decode(order, letters) == want
