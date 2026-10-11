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

import threading
from typing import Any, Mapping, Optional, Sequence

import torch

from .model import PairEncoder, PusulaModel
from .schema import Question, answer_from_probs
from .taxonomy import Q_DOMAIN, Q_INTENT, Q_SUBDOMAIN, Taxonomy, load_taxonomy

QuestionsIn = Mapping[str, Question | Mapping[str, Any]]


def _as_questions(questions: QuestionsIn) -> dict[str, Question]:
    return {k: q if isinstance(q, Question) else Question(**q) for k, q in questions.items()}


class BaseEngine:
    """decide() and route() on top of a subclass's decide_many()."""

    model_name = "pusula"

    def decide_many(self, requests: Sequence[tuple[str, QuestionsIn]]) -> list[dict[str, dict[str, Any]]]:
        raise NotImplementedError

    def decide(self, state: str, questions: QuestionsIn) -> dict[str, dict[str, Any]]:
        return self.decide_many([(state, questions)])[0]

    def route(self, state: str, taxonomy: Optional[Taxonomy] = None, universal: bool = True) -> dict[str, Any]:
        """Hierarchical routing alan -> alt_alan -> niyet (three passes), plus the universal questions."""
        tax = taxonomy or load_taxonomy()
        first = {Q_DOMAIN: tax.domain_question()}
        if universal:
            first.update(tax.universal)
        answers = self.decide(state, first)
        domain = answers[Q_DOMAIN]["choice"]
        answers.update(self.decide(state, {Q_SUBDOMAIN: tax.subdomain_question(domain)}))
        sub = answers[Q_SUBDOMAIN]["choice"]
        answers.update(self.decide(state, {Q_INTENT: tax.intent_question(domain, sub)}))
        return {
            "path": f"{domain}.{sub}.{answers[Q_INTENT]['choice']}",
            "answers": answers,
        }


class D3Engine(BaseEngine):
    """Pusula's main model: Decision 3.0 fine-tuned for Turkish (4-bit base + LoRA adapter)."""

    model_name = "pusula-d3"

    def __init__(self, decider, token_budget: int = 16384):
        self.decider = decider
        self.token_budget = token_budget
        self._lock = threading.Lock()

    @classmethod
    def load(cls, base: str, adapter: Optional[str] = None, device: str = "cuda:0") -> "D3Engine":
        from .d3model import D3Decider

        decider = D3Decider.load(base, adapter=adapter, device=device)
        decider.backbone.eval()
        engine = cls(decider)
        if adapter is None:
            engine.model_name = "d3-base"
        return engine

    @torch.inference_mode()
    def decide_many(self, requests: Sequence[tuple[str, QuestionsIn]]) -> list[dict[str, dict[str, Any]]]:
        from .d3model import render

        jobs = [(r, name, q) for r, (state, qs) in enumerate(requests) for name, q in _as_questions(qs).items()]
        d = self.decider
        ids = d.encode([render(d.tokenizer, requests[r][0], q, d.codes) for r, _, q in jobs])
        results: list[dict[str, dict[str, Any]]] = [{} for _ in requests]
        order = sorted(range(len(jobs)), key=lambda i: len(ids[i]))
        with self._lock:
            start = 0
            while start < len(order):
                end, width = start, 0
                while end < len(order) and max(width, len(ids[order[end]])) * (end - start + 1) <= self.token_budget:
                    width = max(width, len(ids[order[end]]))
                    end += 1
                end = max(end, start + 1)
                batch = order[start:end]
                counts = [len(jobs[i][2].option_keys()) for i in batch]
                with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=d.device.type == "cuda"):
                    logits = d.logits([ids[i] for i in batch], counts)
                probs = torch.softmax(logits.float(), dim=-1).cpu()
                for row, i in enumerate(batch):
                    r, name, q = jobs[i]
                    results[r][name] = answer_from_probs(q, probs[row, : counts[row]].tolist())
                start = end
        return results


class DecisionEngine(BaseEngine):
    """Small and fast: a TabiBERT cross-encoder distilled from the Turkish decision data."""
    def __init__(self, model: PusulaModel, tokenizer, max_length: int = 1024,
                 device: Optional[torch.device] = None, max_pairs_per_pass: int = 256):
        self.model = model
        self.enc = PairEncoder(tokenizer, max_length=max_length)
        self.device = device or next(model.parameters()).device
        self.max_pairs = max_pairs_per_pass
        self._option_cache: dict[str, list[list[int]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> "DecisionEngine":
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if dtype is None and dev.type == "cuda":
            dtype = torch.bfloat16
        model, tokenizer, cfg = PusulaModel.load(path, device=dev, dtype=dtype)
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
