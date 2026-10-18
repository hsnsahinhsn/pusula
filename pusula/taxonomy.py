"""Hierarchical decision taxonomy: alan (domain) -> alt_alan (subdomain) -> niyet (intent),
plus universal questions asked of every state. Source of truth: ``pusula/data/taxonomy.yaml``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import yaml

from .schema import Question

DEFAULT_PATH = Path(__file__).parent / "data" / "taxonomy.yaml"
KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_SIBLINGS = 25

# Question names used for hierarchical routing.
Q_DOMAIN, Q_SUBDOMAIN, Q_INTENT = "alan", "alt_alan", "niyet"
ROUTING_QUESTIONS = (Q_DOMAIN, Q_SUBDOMAIN, Q_INTENT)
# Question name for the first level of an institution's own routing list (the second level is Q_INTENT).
Q_TOPIC = "konu"


def universal_subset(tax: "Taxonomy", universal) -> dict[str, Question]:
    """True -> every universal question, False/None -> none, a list of names -> only those (ValueError for unknown names)."""
    if universal is True:
        return dict(tax.universal)
    if not universal:
        return {}
    if isinstance(universal, str) or not all(isinstance(n, str) for n in universal):
        raise ValueError("universal must be true, false or a list of question names")
    unknown = [n for n in universal if n not in tax.universal]
    if unknown:
        raise ValueError(f"unknown universal questions: {unknown}; choose from {list(tax.universal)}")
    return {n: tax.universal[n] for n in universal}


def topic_questions(topics: dict) -> tuple[Question, dict[str, Question]]:
    """An institution's own routing list -> (topic question, intent question per topic that has intents).

    ``topics`` maps a topic key to its description, or to ``{"description": ..., "intents": {key: description}}``:
        {"kardiyoloji": {"description": "Kalp ve damar hastalıkları",
                         "intents": {"randevu": "Kardiyoloji randevusu", "sonuc": "Tetkik sonucu sorusu"}},
         "faturalama": "Ödeme, fatura ve sigorta işlemleri"}
    Raises ValueError for an invalid list (fewer than 2 or more than the allowed number of options, empty keys).
    """
    if not isinstance(topics, dict):
        raise ValueError("topics must map topic keys to descriptions")
    criteria: dict[str, Optional[str]] = {}
    intents: dict[str, Question] = {}
    for key, spec in topics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("topic keys must be non-empty strings")
        if spec is None or isinstance(spec, str):
            criteria[key] = spec or None
            continue
        if not isinstance(spec, dict) or set(spec) - {"description", "intents"}:
            raise ValueError(f"topic {key!r}: use a description or {{'description': ..., 'intents': {{...}}}}")
        criteria[key] = spec.get("description") or None
        if spec.get("intents"):
            intents[key] = Question(type="choice",
                                    instructions=f"{criteria[key] or key} konusunda kişi tam olarak ne istiyor?",
                                    criteria=dict(spec["intents"]))
    return Question(type="choice", instructions="Bu mesaj hangi konuya ait?", criteria=criteria), intents


@dataclass(frozen=True)
class Subdomain:
    key: str
    name: str
    description: str
    intents: dict[str, str]


@dataclass(frozen=True)
class Domain:
    key: str
    name: str
    description: str
    subdomains: dict[str, Subdomain]


@dataclass(frozen=True)
class IntentPath:
    domain: str
    subdomain: str
    intent: str

    def as_labels(self) -> dict[str, str]:
        return {Q_DOMAIN: self.domain, Q_SUBDOMAIN: self.subdomain, Q_INTENT: self.intent}

    def __str__(self) -> str:
        return f"{self.domain}.{self.subdomain}.{self.intent}"


@dataclass
class Taxonomy:
    version: int
    domains: dict[str, Domain]
    universal: dict[str, Question]
    agent: dict[str, Question] = field(default_factory=dict)  # AI agent decisions (ajan soruları)
    _questions: dict = field(default_factory=dict, repr=False)

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Taxonomy":
        raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
        domains = {
            dk: Domain(
                key=dk,
                name=d["name"],
                description=d["description"],
                subdomains={
                    sk: Subdomain(sk, s["name"], s["description"], dict(s["intents"]))
                    for sk, s in d["subdomains"].items()
                },
            )
            for dk, d in raw["domains"].items()
        }
        universal = {name: Question(**q) for name, q in raw["universal"].items()}
        agent = {name: Question(**q) for name, q in (raw.get("agent") or {}).items()}
        tax = cls(version=int(raw["version"]), domains=domains, universal=universal, agent=agent)
        tax.validate()
        return tax

    def validate(self) -> None:
        def check_keys(keys, where):
            keys = list(keys)
            if not 2 <= len(keys) <= MAX_SIBLINGS and where != "intents of a subdomain":
                raise ValueError(f"{where}: needs 2-{MAX_SIBLINGS} entries, has {len(keys)}")
            for k in keys:
                if not KEY_RE.match(k):
                    raise ValueError(f"{where}: invalid key {k!r}")

        check_keys(self.domains, "domains")
        for d in self.domains.values():
            check_keys(d.subdomains, f"subdomains of {d.key}")
            for s in d.subdomains.values():
                if not 2 <= len(s.intents) <= MAX_SIBLINGS:
                    raise ValueError(f"intents of {d.key}.{s.key}: needs 2-{MAX_SIBLINGS}")
                check_keys(s.intents, "intents of a subdomain")
        clash = set(self.universal) & set(ROUTING_QUESTIONS)
        if clash:
            raise ValueError(f"universal question names clash with routing: {clash}")
        clash = set(self.agent) & (set(self.universal) | set(ROUTING_QUESTIONS))
        if clash:
            raise ValueError(f"agent question names clash with universal or routing questions: {clash}")

    # ---- traversal -----------------------------------------------------------------------
    def intent_paths(self) -> Iterator[IntentPath]:
        for d in self.domains.values():
            for s in d.subdomains.values():
                for i in s.intents:
                    yield IntentPath(d.key, s.key, i)

    def path(self, domain: str, subdomain: str, intent: str) -> IntentPath:
        s = self.domains[domain].subdomains[subdomain]
        if intent not in s.intents:
            raise KeyError(f"{domain}.{subdomain}.{intent}")
        return IntentPath(domain, subdomain, intent)

    def describe(self, p: IntentPath) -> str:
        d = self.domains[p.domain]
        s = d.subdomains[p.subdomain]
        return f"{d.name} > {s.name} > {s.intents[p.intent]}"

    # ---- questions -----------------------------------------------------------------------
    def domain_question(self) -> Question:
        if "domain" not in self._questions:
            self._questions["domain"] = Question(
                type="choice",
                instructions="Bu mesaj hangi ana alana ait?",
                criteria={d.key: f"{d.name}: {d.description}" for d in self.domains.values()},
            )
        return self._questions["domain"]

    def subdomain_question(self, domain: str) -> Question:
        key = ("sub", domain)
        if key not in self._questions:
            d = self.domains[domain]
            self._questions[key] = Question(
                type="choice",
                instructions=f"{d.name} alanında bu mesaj hangi alt alana ait?",
                criteria={s.key: f"{s.name}: {s.description}" for s in d.subdomains.values()},
            )
        return self._questions[key]

    def intent_question(self, domain: str, subdomain: str) -> Question:
        key = ("intent", domain, subdomain)
        if key not in self._questions:
            s = self.domains[domain].subdomains[subdomain]
            self._questions[key] = Question(
                type="choice",
                instructions=f"{s.name} konusunda kullanıcının niyeti nedir?",
                criteria=dict(s.intents),
            )
        return self._questions[key]

    def questions_for(self, p: IntentPath, universal: bool = True) -> dict[str, Question]:
        qs = {
            Q_DOMAIN: self.domain_question(),
            Q_SUBDOMAIN: self.subdomain_question(p.domain),
            Q_INTENT: self.intent_question(p.domain, p.subdomain),
        }
        if universal:
            qs.update(self.universal)
        return qs

    def to_dict(self) -> dict:
        """Plain JSON-able form (served by the API and embedded in the labeling artifact)."""
        return {
            "version": self.version,
            "universal": {k: q.model_dump(exclude_none=True) for k, q in self.universal.items()},
            "agent": {k: q.model_dump(exclude_none=True) for k, q in self.agent.items()},
            "domains": {
                d.key: {
                    "name": d.name,
                    "description": d.description,
                    "subdomains": {
                        s.key: {"name": s.name, "description": s.description, "intents": s.intents}
                        for s in d.subdomains.values()
                    },
                }
                for d in self.domains.values()
            },
        }


@lru_cache(maxsize=4)
def load_taxonomy(path: Optional[str] = None) -> Taxonomy:
    return Taxonomy.load(path)
