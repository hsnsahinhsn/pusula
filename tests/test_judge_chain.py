"""Pipelined judge passes (--chain-outs, 30 Oct 2026) must give the same rows as three separate passes."""
import hashlib
import importlib.util
import json
from pathlib import Path

from pusula import judge
from pusula.jsonl import read_jsonl
from pusula.schema import Question
from pusula.taxonomy import load_taxonomy

LABEL20 = Path(__file__).resolve().parents[1] / "runs" / "api" / "v20" / "label20.py"


def _label20():
    spec = importlib.util.spec_from_file_location("label20_t", LABEL20)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_llm(monkeypatch, fail=lambda prompt, names: False):
    async def chat_json(client, model, messages, schema, **kw):
        prompt = messages[-1]["content"]
        if fail(prompt, set(schema["properties"])):
            return None
        out = {}
        for name, spec in schema["properties"].items():
            h = int(hashlib.sha256((prompt + name).encode()).hexdigest(), 16)
            out[name] = spec["enum"][h % len(spec["enum"])]
        return out

    async def no_wait(url):
        return None

    monkeypatch.setattr(judge, "chat_json", chat_json)
    monkeypatch.setattr(judge, "wait_for_server", no_wait)
    monkeypatch.setattr(judge, "make_client", lambda url: None)


def _p1(tmp_path, label20):
    tax = load_taxonomy()
    own = {"s1": Question(type="noul", instructions="Kişi iade istiyor mu?").model_dump(exclude_none=True)}
    fab = [{"id": f"f{i}", "state": f"Mesaj {i}: kargom gelmedi, paramı geri istiyorum ({i}).", "questions": own,
            "labels": {"generator": {"s1": "true"}}, "meta": {"channel": "e-posta"}} for i in range(12)]
    old = [{"id": "o1", "state": "Eski mesaj, faturam yanlış.", "questions": {
        "alan": tax.domain_question().model_dump(exclude_none=True)}, "labels": {"final": {"alan": "ticaret_eticaret"}}}]
    for name, rows in (("fab.jsonl", fab), ("old.jsonl", old)):
        (tmp_path / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    p1 = tmp_path / "p1.jsonl"
    label20.prepare([str(tmp_path / "fab.jsonl")], [str(tmp_path / "old.jsonl")], str(p1), 1)
    return p1


def _rows(path):
    return sorted(((r["id"], sorted(r["questions"]), json.dumps(r["labels"], sort_keys=True)) for r in read_jsonl(path)))


COMMON = ["--layout", "questions-first", "--fixed-order", "--concurrency", "3"]


def _sequential(tmp_path, p1, label20):
    j1, p2, j2, p3, j3 = (tmp_path / f"s_{n}.jsonl" for n in ("j1", "p2", "j2", "p3", "j3"))
    judge.main(["--inp", str(p1), "--out", str(j1), "--source", "q1", *COMMON])
    label20.advance(str(j1), str(p2))
    judge.main(["--inp", str(p2), "--out", str(j2), "--source", "q2", "--questions", "alt_alan", *COMMON])
    label20.advance(str(j2), str(p3))
    judge.main(["--inp", str(p3), "--out", str(j3), "--source", "q3", "--questions", "niyet", *COMMON])
    return j1, j2, j3


def _chain(tmp_path, p1):
    c1, c2, c3 = (str(tmp_path / f"c_{n}.jsonl") for n in ("j1", "j2", "j3"))
    judge.main(["--inp", str(p1), "--out", c1 + ".p0", "--source", "q1", "--chain-outs", f"{c2},{c3}",
                "--chain-sources", "q2,q3", "--advance-py", str(LABEL20), *COMMON])
    return c1 + ".p0", c2 + ".p0", c3 + ".p0"


def test_chain_matches_separate_passes(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    label20 = _label20()
    p1 = _p1(tmp_path, label20)
    seq = _sequential(tmp_path, p1, label20)
    chain = _chain(tmp_path, p1)
    for s, c in zip(seq, chain):
        assert _rows(s) and _rows(s) == _rows(c), (s, c)
    assert len(_rows(seq[2])) > 0  # the third pass was reached


def test_chain_resumes_from_its_part_files(tmp_path, monkeypatch):
    label20 = _label20()
    p1 = _p1(tmp_path, label20)
    _fake_llm(monkeypatch)
    seq = _sequential(tmp_path, p1, label20)
    # first run: every alt_alan answer fails, so the chain stops after pass 1
    _fake_llm(monkeypatch, fail=lambda prompt, names: names == {"alt_alan"})
    c1, c2, c3 = _chain(tmp_path, p1)
    assert _rows(c1) == _rows(seq[0]) and not _rows(c2) and not _rows(c3)
    # second run: pass 1 is done; the chain continues from its own pass 1 part
    _fake_llm(monkeypatch)
    _chain(tmp_path, p1)
    for s, c in zip(seq, (c1, c2, c3)):
        assert _rows(s) == _rows(c)
    assert len(list(read_jsonl(c1))) == len(_rows(seq[0]))  # nothing asked twice


def test_chain_paths_follow_the_card_suffix():
    assert judge.chain_paths("/a/new_j1.jsonl.p3", ["/a/new_j2.jsonl", "/a/new_j3.jsonl"]) == \
        ["/a/new_j2.jsonl.p3", "/a/new_j3.jsonl.p3"]
    assert judge.chain_paths("/a/j1.jsonl", ["/a/j2.jsonl"]) == ["/a/j2.jsonl"]


def test_chain_skips_answers_in_other_cards_parts(tmp_path, monkeypatch):
    """A later pass answered on another card (e.g. by a separate pass before a restart) is not asked again."""
    label20 = _label20()
    p1 = _p1(tmp_path, label20)
    _fake_llm(monkeypatch, fail=lambda prompt, names: names == {"alt_alan"})
    c1, c2, c3 = _chain(tmp_path, p1)
    first = next(iter(read_jsonl(c1)))
    other = tmp_path / "c_j2.jsonl.p1"   # another card's q2 part holds this message
    other.write_text(json.dumps({**first, "labels": {**first["labels"], "q2": {"alt_alan": "x"}}}) + "\n")
    asked = []
    _fake_llm(monkeypatch, fail=lambda prompt, names: asked.append(names) and False)
    _chain(tmp_path, p1)
    assert first["id"] not in {r["id"] for r in read_jsonl(c2)}
    assert len(list(read_jsonl(c2))) == len(list(read_jsonl(c1))) - 1
