"""Pusula student model: a TabiBERT cross-encoder that scores (question + option, state) pairs.

Each option of a question becomes one input ``[CLS] question/option [SEP] state [SEP]``; the encoder's
mean-pooled output goes through a linear scorer, and a softmax over a question's options gives its
answer distribution. Options are read as text, so any option set works at inference time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .schema import QUESTION_TYPES, Question

BASE_MODEL = "boun-tabilab/TabiBERT"
FORMAT_VERSION = 1


def default_attn_implementation() -> str:
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


class PairEncoder:
    """Builds token ids for (question + option, state) pairs without re-tokenizing the state per option."""

    def __init__(self, tokenizer, max_length: int = 1024, max_option_tokens: int = 160):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_option_tokens = max_option_tokens
        self.cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if self.cls_id is None or self.sep_id is None:
            raise ValueError("tokenizer needs CLS/SEP (or BOS/EOS) tokens")

    def _ids(self, texts: Sequence[str]) -> list[list[int]]:
        return self.tok(list(texts), add_special_tokens=False)["input_ids"]

    def encode_states(self, states: Sequence[str]) -> list[list[int]]:
        return [ids[: self.max_length] for ids in self._ids(states)]

    @staticmethod
    def option_text(question: Question, option_text: str) -> str:
        return f"Soru: {question.instructions}\nSeçenek: {option_text}"

    def encode_options(self, question: Question) -> list[list[int]]:
        texts = [self.option_text(question, t) for t in question.option_texts()]
        return [ids[: self.max_option_tokens] for ids in self._ids(texts)]

    def join(self, option_ids: list[int], state_ids: list[int]) -> list[int]:
        room = self.max_length - len(option_ids) - 3
        return [self.cls_id, *option_ids, self.sep_id, *state_ids[: max(room, 0)], self.sep_id]

    def pad(self, seqs: Sequence[Sequence[int]], device=None) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids.to(device), mask.to(device)


class PusulaModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(encoder.config.hidden_size, 1)
        self.temperature = {t: 1.0 for t in QUESTION_TYPES}
        self.meta: dict = {}

    @classmethod
    def from_base(cls, name: str = BASE_MODEL, attn_implementation: Optional[str] = None,
                  dropout: float = 0.1) -> "PusulaModel":
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            name, attn_implementation=attn_implementation or default_attn_implementation())
        model = cls(encoder, dropout)
        model.meta = {"base_model": name}
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        pooled = pooled.to(self.scorer.weight.dtype)
        return self.scorer(self.dropout(pooled)).squeeze(-1).float()

    # ---- persistence ---------------------------------------------------------------------
    def save(self, path: str | Path, tokenizer=None, max_length: int = 1024) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(path / "encoder")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.scorer.state_dict().items()},
                  str(path / "scorer.safetensors"))
        cfg = {
            "format_version": FORMAT_VERSION,
            "temperature": self.temperature,
            "max_length": max_length,
            **self.meta,
        }
        (path / "pusula.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu",
             dtype: Optional[torch.dtype] = None, attn_implementation: Optional[str] = None):
        """Returns (model, tokenizer, config)."""
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        cfg = json.loads((path / "pusula.json").read_text(encoding="utf-8"))
        if cfg.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format {cfg.get('format_version')}")
        encoder = AutoModel.from_pretrained(
            path / "encoder", attn_implementation=attn_implementation or default_attn_implementation(),
            dtype=dtype)
        model = cls(encoder)
        model.scorer.load_state_dict(load_file(str(path / "scorer.safetensors")))
        model.temperature = {t: float(cfg["temperature"].get(t, 1.0)) for t in QUESTION_TYPES}
        model.meta = {k: v for k, v in cfg.items() if k not in ("format_version", "temperature")}
        tokenizer = AutoTokenizer.from_pretrained(path / "encoder")
        model.to(device)
        if dtype is not None:
            model.scorer.float()  # keep the scorer in fp32 for stable logits
        model.eval()
        return model, tokenizer, cfg
