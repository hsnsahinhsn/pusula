"""High-level inference: answer d3-style questions about a state, or route it through the taxonomy.

    from pusula.inference import DecisionEngine
    engine = DecisionEngine.load("runs/full/model/best")
    engine.decide("Kartımdan bilgim dışında 4.500 TL çekilmiş!", {
        "ekip": {"type": "choice", "instructions": "Hangi ekip ilgilenmeli?",
                 "criteria": {"dolandiricilik": "Şüpheli işlem ve fraud", "kart": "Kart işlemleri"}},
        "acil": {"type": "noul", "instructions": "Hemen müdahale gerekiyor mu?"},
    })
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping, Optional, Sequence

import torch

from .model import PairEncoder, PusulaModel
from .schema import Question, answer_from_probs
from .taxonomy import (Q_DOMAIN, Q_INTENT, Q_SUBDOMAIN, Q_TOPIC, Taxonomy, load_taxonomy, topic_questions,
                       universal_subset)

QuestionsIn = Mapping[str, Question | Mapping[str, Any]]


def _as_questions(questions: QuestionsIn) -> dict[str, Question]:
    return {k: q if isinstance(q, Question) else Question(**q) for k, q in questions.items()}


_CUTS = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")


def split_text(text: str, window_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split ``text`` into parts of at most ``window_chars`` characters, cutting at paragraph, line, sentence or word
    boundaries where possible. Consecutive parts share up to ``overlap_chars`` characters, so that a sentence on a
    boundary is also seen whole. Every character of ``text`` is in at least one part."""
    if window_chars < 2:
        raise ValueError("window_chars must be at least 2")
    overlap_chars = max(0, min(overlap_chars, window_chars // 2))
    if len(text) <= window_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = start + window_chars
        if end >= len(text):
            parts.append(text[start:])
            break
        lo = start + window_chars // 2
        cut = next((i + len(sep) for sep in _CUTS if (i := text.rfind(sep, lo, end - len(sep) + 1)) >= 0), end)
        parts.append(text[start:cut])
        nxt = cut - overlap_chars
        if overlap_chars:
            sp = text.find(" ", nxt, cut)
            nxt = sp + 1 if sp >= 0 else nxt
        start = max(nxt, start + 1)
    return parts


def _finding(answer: Mapping[str, Any]) -> float:
    """How far an answer is from the neutral one: P(true) for noul, the expected level for score, and one minus the
    probability of the first option (for example ``guvenli``) for choice."""
    if answer["type"] == "noul":
        return answer["noul"]
    if answer["type"] == "score":
        return answer["score"]
    return 1.0 - next(iter(answer["probabilities"].values()))


class BaseEngine:
    """decide(), route() and scan() on top of a subclass's decide_many()."""

    model_name = "pusula"
    default_window_chars = 12000

    def _fits(self, part: str) -> bool:
        """Whether a part is read whole in one pass (subclasses with a context limit override this)."""
        return True

    def _parts(self, state: str, window_chars: int, overlap_chars: int) -> list[str]:
        parts = []
        for part in split_text(state, window_chars, overlap_chars):
            if self._fits(part) or len(part) < 64:
                parts.append(part)
            else:
                parts += self._parts(part, len(part) // 2 + 1, overlap_chars)
        return parts

    def scan(self, state: str, questions: QuestionsIn, window_chars: Optional[int] = None,
             overlap_chars: int = 200) -> dict[str, Any]:
        """Check a long text (a web page, a document, an e-mail thread) part by part.

        The text is split into parts that the model reads whole; every part is answered, and for every question the
        answer of the part with the strongest finding is returned, together with its index in ``part``: the highest
        probability of *true* for ``noul``, the highest expected level for ``score`` and the lowest probability of the
        first option for ``choice``. The first option should therefore be the neutral one (as ``guvenli`` is in the
        built-in ``guvenlik`` question). A text that fits in one part is answered exactly as with decide()."""
        qs = _as_questions(questions)
        parts = self._parts(state, window_chars or self.default_window_chars, overlap_chars)
        results = self.decide_many([(p, qs) for p in parts])
        answers = {}
        for name in qs:
            best = max(range(len(parts)), key=lambda i: _finding(results[i][name]))
            answers[name] = {**results[best][name], "part": best}
        return {"answers": answers, "parts": len(parts)}

    def decide_many(self, requests: Sequence[tuple[str, QuestionsIn]]) -> list[dict[str, dict[str, Any]]]:
        raise NotImplementedError

    def decide(self, state: str, questions: QuestionsIn) -> dict[str, dict[str, Any]]:
        return self.decide_many([(state, questions)])[0]

    def route(self, state: str, taxonomy: Optional[Taxonomy] = None, universal: bool | Sequence[str] = True,
              topics: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Hierarchical routing alan -> alt_alan -> niyet (three passes), plus the universal questions.

        ``universal``: True asks every universal question, False none, a list of names only those (fewer questions,
        faster answers). With ``topics`` (an institution's own list, see ``taxonomy.topic_questions``) the route is
        konu -> niyet instead: one pass for the topic and, when that topic lists intents, one pass for the intent."""
        tax = taxonomy or load_taxonomy()
        extra = universal_subset(tax, universal)
        if topics is not None:
            return self._route_topics(state, topics, extra)
        first = {Q_DOMAIN: tax.domain_question(), **extra}
        answers = self.decide(state, first)
        domain = answers[Q_DOMAIN]["choice"]
        answers.update(self.decide(state, {Q_SUBDOMAIN: tax.subdomain_question(domain)}))
        sub = answers[Q_SUBDOMAIN]["choice"]
        answers.update(self.decide(state, {Q_INTENT: tax.intent_question(domain, sub)}))
        return {
            "path": f"{domain}.{sub}.{answers[Q_INTENT]['choice']}",
            "answers": answers,
        }

    def _route_topics(self, state: str, topics: Mapping[str, Any], extra: dict[str, Question]) -> dict[str, Any]:
        topic_q, intent_qs = topic_questions(dict(topics))
        answers = self.decide(state, {Q_TOPIC: topic_q, **extra})
        topic = answers[Q_TOPIC]["choice"]
        if topic not in intent_qs:
            return {"path": topic, "answers": answers}
        answers.update(self.decide(state, {Q_INTENT: intent_qs[topic]}))
        return {"path": f"{topic}.{answers[Q_INTENT]['choice']}", "answers": answers}


class D3Engine(BaseEngine):
    """Pusula's main model: Decision 3.0 fine-tuned for Turkish (4-bit base + LoRA adapter)."""

    model_name = "pusula-d3"

    def __init__(self, decider, token_budget: int = 16384, calibration: Optional[Mapping[str, Any]] = None,
                 own_base: bool = False):
        self.decider = decider
        self.token_budget = token_budget
        self._lock = threading.Lock()
        # own_base: a question that is not one of Pusula's built-in questions (routing, template and agent questions)
        # is answered by the base Decision 3.0 (adapter off). 31 Oct 2026 measurement: on users' own questions in
        # unseen sectors the base model scored higher than the fine-tuned adapters (92.8 vs 90.5 / 91.2).
        self.own_base = own_base
        self._builtin: Optional[tuple[set[str], list[Question]]] = None
        self._base_templates: Optional[list[Question]] = None
        # per-option log-prob biases for the built-in universal questions (pusula.calibrate); applied only when
        # a request asks exactly that question, never to a custom question that happens to share its name
        self._calibration: dict[str, tuple[Question, list[float]]] = {}
        if calibration:
            universal = load_taxonomy().universal
            for name, c in calibration.get("questions", {}).items():
                q = universal.get(name)
                if q is not None and len(q.option_keys()) == c["options"]:
                    self._calibration[name] = (q, list(c["bias"]))

    # Built-in templates that own_base also sends to the base model (user decision, 31 Oct 2026). eksik_bilgi was not
    # in the v1.0 training; on the missing-information exam set the base model scored 98.0 and the v2.0 adapter 88.9
    # (on the real customer messages the adapter scored higher, 95.4 vs 75.6).
    BASE_TEMPLATES = ("eksik_bilgi",)

    @classmethod
    def load(cls, base: str, adapter: Optional[str] = None, device: str = "cuda:0",
             load_in_4bit: Optional[bool] = None, own_base: Optional[bool] = None) -> "D3Engine":
        """`load_in_4bit=None` picks BF16 when the weights fit the GPU, else 4-bit (PUSULA_D3_4BIT overrides).
        `own_base=None` reads PUSULA_OWN_BASE (default "1": own questions are answered by the base model; "0" = adapter)."""
        from .d3model import D3Decider

        base = os.path.expanduser(base)
        adapter = os.path.expanduser(adapter) if adapter else None
        decider = D3Decider.load(base, adapter=adapter, device=device, load_in_4bit=load_in_4bit)
        decider.backbone.eval()
        calib_file = os.path.join(adapter or base, "calibration.json")
        calibration = None
        if os.path.isfile(calib_file):
            with open(calib_file, encoding="utf-8") as f:
                calibration = json.load(f)
        if own_base is None:
            own_base = os.environ.get("PUSULA_OWN_BASE", "1") != "0"   # v2.0 default (user decision, 31 Oct 2026)
        engine = cls(decider, calibration=calibration, own_base=bool(own_base and adapter))
        if adapter is None:
            engine.model_name = "d3-base"
        return engine

    def _is_builtin(self, name: str, q: Question) -> bool:
        """True for a question the adapter was trained on: a routing question (alan, alt_alan, niyet, and konu of an
        institution's own list), a universal (template) question or an agent question, by name or by content. By name
        too, because a built-in question may arrive in an earlier wording: the published benchmark and older clients
        use v1.0 texts and option lists of duygu, guvenlik, kisisel_veri and the routing questions."""
        if self._builtin is None:
            tax = load_taxonomy()
            qs = [tax.domain_question()]
            for d, dom in tax.domains.items():
                qs.append(tax.subdomain_question(d))
                qs += [tax.intent_question(d, sk) for sk in dom.subdomains]
            names = {Q_DOMAIN, Q_SUBDOMAIN, Q_INTENT, Q_TOPIC, *tax.universal, *tax.agent}
            self._builtin = (names, qs + list(tax.universal.values()) + list(tax.agent.values()))
        names, qs = self._builtin
        return name in names or q in qs

    def _use_base(self, name: str, q: Question) -> bool:
        """own_base routing: own questions, and the BASE_TEMPLATES (by name or content), go to the base model."""
        if not self.own_base:
            return False
        if self._base_templates is None:
            universal = load_taxonomy().universal
            self._base_templates = [universal[n] for n in self.BASE_TEMPLATES if n in universal]
        return not self._is_builtin(name, q) or name in self.BASE_TEMPLATES or q in self._base_templates

    @torch.inference_mode()
    def decide_many(self, requests: Sequence[tuple[str, QuestionsIn]]) -> list[dict[str, dict[str, Any]]]:
        from .d3model import render

        jobs = [(r, name, q) for r, (state, qs) in enumerate(requests) for name, q in _as_questions(qs).items()]
        d = self.decider
        ids = d.encode([render(d.tokenizer, requests[r][0], q, d.codes) for r, _, q in jobs])
        results: list[dict[str, dict[str, Any]]] = [{} for _ in requests]
        use_base = [self._use_base(name, q) for _, name, q in jobs]
        with self._lock:
            for base in (False, True):  # one forward pass never mixes adapter and base answers
                order = sorted((i for i in range(len(jobs)) if use_base[i] == base), key=lambda i: len(ids[i]))
                lengths = [len(ids[i]) for i in order]
                start = 0
                while start < len(order):
                    end = self._batch_end(lengths, start)
                    batch = order[start:end]
                    counts = [len(jobs[i][2].option_keys()) for i in batch]
                    with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=d.device.type == "cuda"):
                        logits = d.logits([ids[i] for i in batch], counts, base=True) if base else \
                            d.logits([ids[i] for i in batch], counts)
                    probs = torch.softmax(logits.float(), dim=-1).cpu()
                    for row, i in enumerate(batch):
                        r, name, q = jobs[i]
                        p = probs[row, : counts[row]].tolist()
                        results[r][name] = answer_from_probs(q, p if base else self._calibrate(name, q, p))
                    start = end
        return results

    def _batch_end(self, lengths: Sequence[int], start: int, max_padding: float = 0.25) -> int:
        """End of the forward pass that starts at ``start`` (``lengths`` sorted ascending): as many prompts as fit the
        token budget, stopping before padding exceeds ``max_padding`` of the real tokens. Padding is paid for in full;
        on a CPU, one long prompt padded together with several short ones can make a pass twice as slow."""
        end, real = start + 1, lengths[start]
        while end < len(lengths):
            n, width = end - start + 1, lengths[end]  # the next prompt is the longest so far
            if width * n > self.token_budget or width * n > (real + width) * (1 + max_padding):
                break
            real += width
            end += 1
        return end

    def _calibrate(self, name: str, q: Question, probs: list[float]) -> list[float]:
        from .calibrate import apply_bias

        c = self._calibration.get(name)
        return apply_bias(probs, c[1]) if c and c[0] == q else probs


class DecisionEngine(BaseEngine):
    """Pusula Nano, small and fast: a TabiBERT cross-encoder distilled from the Turkish decision data."""
    def __init__(self, model: PusulaModel, tokenizer, max_length: int = 1024,
                 device: Optional[torch.device] = None, max_pairs_per_pass: int = 256):
        self.model = model
        self.enc = PairEncoder(tokenizer, max_length=max_length)
        self.device = device or next(model.parameters()).device
        self.max_pairs = max_pairs_per_pass
        self._option_cache: dict[str, list[list[int]]] = {}
        self._lock = threading.Lock()
        # room for the text next to the longest option; about 3 characters per token in Turkish text
        self.state_tokens = max_length - self.enc.max_option_tokens - 3
        self.default_window_chars = max(256, 3 * self.state_tokens)

    def _fits(self, part: str) -> bool:
        return len(self.enc._ids([part])[0]) <= self.state_tokens

    @classmethod
    def load(cls, path: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> "DecisionEngine":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None and dev.type == "cuda":
            dtype = torch.bfloat16
        model, tokenizer, cfg = PusulaModel.load(os.path.expanduser(path), device=dev, dtype=dtype)
        return cls(model, tokenizer, max_length=int(cfg.get("max_length", 1024)), device=dev)

    @property
    def model_name(self) -> str:
        return self.model.meta.get("name", "pusula")

    def _options(self, q: Question) -> list[list[int]]:
        key = q.model_dump_json()
        ids = self._option_cache.get(key)
        if ids is None:
            ids = self.enc.encode_options(q)
            if len(self._option_cache) < 10_000:
                self._option_cache[key] = ids
        return ids

    @torch.inference_mode()
    def _logits(self, seqs: list[list[int]]) -> torch.Tensor:
        outs = []
        for i in range(0, len(seqs), self.max_pairs):
            ids, mask = self.enc.pad(seqs[i:i + self.max_pairs], self.device)
            outs.append(self.model(ids, mask).float().cpu())
        return torch.cat(outs) if outs else torch.empty(0)

    def decide_many(self, requests: Sequence[tuple[str, QuestionsIn]]) -> list[dict[str, dict[str, Any]]]:
        """Answer several (state, questions) requests in as few forward passes as possible."""
        parsed = [(state, _as_questions(qs)) for state, qs in requests]
        state_ids = self.enc.encode_states([s for s, _ in parsed])
        seqs, spans = [], []
        for r, (_, qs) in enumerate(parsed):
            for name, q in qs.items():
                opts = self._options(q)
                spans.append((r, name, q, len(seqs), len(opts)))
                seqs += [self.enc.join(o, state_ids[r]) for o in opts]
        with self._lock:
            logits = self._logits(seqs)
        results: list[dict[str, dict[str, Any]]] = [{} for _ in parsed]
        for r, name, q, start, n in spans:
            t = self.model.temperature.get(q.type, 1.0)
            probs = torch.softmax(logits[start:start + n] / t, dim=-1).tolist()
            results[r][name] = answer_from_probs(q, probs)
        return results
