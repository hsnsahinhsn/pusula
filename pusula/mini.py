"""Pusula Mini: a small d3-format decision model (Qwen3.5-0.8B backbone + code readout).

`init` turns a small Qwen3.5 checkpoint into a d3-format directory that `pusula.d3train` can train with
`--full` and that `D3Engine` / `pusula.d3train eval` load like d3 itself. The code readout starts from the
model's own (tied) token embeddings of the answer codes, i.e. its next-token preference for each code.

    python -m pusula.mini init --base Qwen/Qwen3.5-0.8B --d3 ~/d3 --out runs/full/mini_init
    torchrun --nproc-per-node 2 -m pusula.d3train train --full --d3 runs/full/mini_init \
        --data runs/full/curated --out runs/full/mini --max-prompts 0 --lr 3e-5 --readout-lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .d3model import D3_FORMAT_FILES
from .llm import console

DEFAULT_BASE = "Qwen/Qwen3.5-0.8B"


def init(base: str, d3_dir: str, out: str) -> Path:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    d3, out_dir = Path(d3_dir), Path(out)
    cfg = json.loads((d3 / "decision_config.json").read_text(encoding="utf-8"))
    ids = cfg["token_ids"]
    ref, own = AutoTokenizer.from_pretrained(d3), AutoTokenizer.from_pretrained(base)
    if ref.convert_ids_to_tokens(ids) != own.convert_ids_to_tokens(ids):
        raise ValueError(f"{base} does not share d3's vocabulary for the answer codes")
    backbone = Qwen3_5Model.from_pretrained(base, dtype=torch.bfloat16)
    readout = backbone.get_input_embeddings().weight[ids].detach().float().contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(str(out_dir))
    save_file({"weight": readout.cpu()}, str(out_dir / "readout.safetensors"))
    for name in D3_FORMAT_FILES:  # d3's prompt contract: same chat template, codes and tokenizer
        if (d3 / name).exists():
            shutil.copy2(d3 / name, out_dir / name)
    (out_dir / "pusula_mini.json").write_text(json.dumps({"base": base, "format_id": cfg["format_id"]}, indent=2),
                                              encoding="utf-8")
    console.print(f"[green]Pusula Mini başlangıcı: {base} + d3 kod okuyucu [{readout.shape[0]}x{readout.shape[1]}] → {out_dir}[/green]")
    return out_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--base", default=DEFAULT_BASE)
    i.add_argument("--d3", required=True, help="local vllm-sr/d3 directory (prompt contract and answer codes)")
    i.add_argument("--out", required=True)
    a = p.parse_args(argv)
    init(a.base, a.d3, a.out)


if __name__ == "__main__":
    main()
