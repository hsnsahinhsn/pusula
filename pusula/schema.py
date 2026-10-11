"""Decision questions and examples, in the d3 (vllm-sr/d3) request/answer contract.

A request is a ``state`` plus named ``questions``. Each question is one of:

- ``choice``: ``criteria`` maps option keys to descriptions (or None).
- ``noul``:   yes/no; optional ``criteria`` describes ``"false"`` and ``"true"``.
- ``score``:  ``criteria`` is an ordered list of 2-10 levels, lowest first.

Answers use d3's shapes, so Pusula is a drop-in replacement for d3's ``/v1/systemone``.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["choice", "noul", "score"]
QUESTION_TYPES: tuple[str, ...] = ("choice", "noul", "score")
NOUL_KEYS = ("false", "true")
NOUL_DEFAULTS = ("Hayır / yanlış", "Evet / doğru")
SCORE_LEVELS = (2, 10)
MAX_OPTIONS = 64


class Question(BaseModel):
    type: QuestionType
    instructions: str = Field(..., min_length=1, max_length=2000)
    criteria: Optional[Union[dict[str, Optional[str]], list[str]]] = None

    @model_validator(mode="after")
    def _check_criteria(self) -> "Question":
        c = self.criteria
        if self.type == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= MAX_OPTIONS:
                raise ValueError(f"choice criteria must map 2-{MAX_OPTIONS} option keys to descriptions")
            if any(not str(k).strip() for k in c):
                raise ValueError("choice option keys must be non-empty")
        elif self.type == "noul":
            if c is not None and (not isinstance(c, dict) or set(c) - set(NOUL_KEYS)):
                raise ValueError('noul criteria may only describe "false" and "true"')
        else:
            low, high = SCORE_LEVELS
            if not isinstance(c, list) or not low <= len(c) <= high:
                raise ValueError(f"score criteria must be an ordered list of {low}-{high} levels")
            if len(set(c)) != len(c) or any(not str(level).strip() for level in c):
                raise ValueError("score levels must be distinct and non-empty")
        return self

    def option_keys(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "noul":
            return list(NOUL_KEYS)
        return [str(i) for i in range(len(self.criteria))]

    def option_texts(self) -> list[str]:
        """Human-readable option texts the student model reads, in option_keys() order."""
        if self.type == "choice":
            return [
                desc if desc else key.replace("_", " ")
                for key, desc in self.criteria.items()
            ]
        if self.type == "noul":
            c = self.criteria or {}
            return [c.get(k) or d for k, d in zip(NOUL_KEYS, NOUL_DEFAULTS)]
        n = len(self.criteria)
        return [f"Seviye {i + 1}/{n}: {level}" for i, level in enumerate(self.criteria)]

    def to_d3(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


class Example(BaseModel):
    """One state with its questions and every label source collected so far.

    ``labels[source][question] = option key``; sources: generator, judge, d3, final, human.
    ``teacher[question]`` = d3 probabilities in ``option_keys()`` order.
    """

    id: str
    state: str = Field(..., min_length=1)
    questions: dict[str, Question]
    meta: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, dict[str, str]] = Field(default_factory=dict)
    teacher: dict[str, list[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_labels(self) -> "Example":
        for source, answers in self.labels.items():
            for name, key in answers.items():
                q = self.questions.get(name)
                if q is None:
                    raise ValueError(f"label {source}.{name} has no question")
                if key not in q.option_keys():
                    raise ValueError(f"label {source}.{name}={key!r} is not an option")
        for name, probs in self.teacher.items():
            q = self.questions.get(name)
            if q is None or len(probs) != len(q.option_keys()):
                raise ValueError(f"teacher probabilities for {name} do not match its options")
        return self


def stable_id(text: str) -> str:
    """Deterministic id from normalized text (dedup key across runs and machines)."""
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def answer_from_probs(question: Question, probs: Sequence[float]) -> dict[str, Any]:
    """d3-shaped answer for one question from probabilities in option_keys() order."""
    values = [float(p) for p in probs]
    total = sum(values)
    if len(values) != len(question.option_keys()) or total <= 0 or any(
        not math.isfinite(v) or v < 0 for v in values
    ):
        raise ValueError("need one finite non-negative probability per option")
    values = [v / total for v in values]
    keys = question.option_keys()
    if question.type == "noul":
        return {"type": "noul", "noul": values[1]}
    best = max(range(len(values)), key=values.__getitem__)
    if question.type == "choice":
        return {
            "type": "choice",
            "choice": keys[best],
            "probabilities": dict(zip(keys, values)),
            "confidence": values[best],
        }
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(values)),
        "probabilities": dict(zip(keys, values)),
        "confidence": values[best],
        "legend": dict(zip(keys, question.criteria)),
    }


def probs_from_d3_answer(question: Question, answer: dict[str, Any]) -> list[float]:
    """Inverse of answer_from_probs for d3's own answers (choice / noul / score)."""
    if "error" in answer:
        raise ValueError(answer.get("message") or answer["error"])
    if question.type == "noul":
        p = float(answer["noul"])
        return [1.0 - p, p]
    probabilities = answer["probabilities"]
    return [float(probabilities[k]) for k in question.option_keys()]


def argmax_key(question: Question, probs: Sequence[float]) -> str:
    keys = question.option_keys()
    return keys[max(range(len(probs)), key=lambda i: probs[i])]
