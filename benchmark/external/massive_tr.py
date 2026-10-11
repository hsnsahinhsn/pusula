"""Turkish slice of Amazon MASSIVE (CC BY 4.0) as a Pusula evaluation set: real, human-written Turkish
utterances, answered as one ad-hoc choice question over MASSIVE's 60 intents (zero-shot for every model).

    python benchmark/external/massive_tr.py --out runs/external/massive_tr_test.jsonl

Source: FitzGerald et al., 2022, "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset
with 51 Typologically-Diverse Languages", https://huggingface.co/datasets/AmazonScience/massive (CC BY 4.0).
The data is downloaded at run time and not redistributed with Pusula.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "AmazonScience/massive"
INSTRUCTIONS = "Kullanıcı sesli asistandan ne istiyor? En uygun niyeti seç."


PARQUET = "refs/convert/parquet"  # the Hub's parquet export of the dataset


def load_split(split: str) -> tuple[list[dict], list[str]]:
    """Rows of the tr-TR split and the intent names (ClassLabel order from the parquet metadata)."""
    import pyarrow.parquet as pq

    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=PARQUET)
             if f.startswith(f"tr-TR/{split}/") and f.endswith(".parquet")]
    rows, names = [], []
    for f in sorted(files):
        table = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset", revision=PARQUET))
        names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        rows += table.select(["id", "utt", "intent"]).to_pylist()
    return rows, names


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows, names = load_split(a.split)
    intent = lambda r: names[r["intent"]]  # noqa: E731
    intents = sorted(names)
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {k: None for k in intents}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            ex = {"id": f"massive-tr-{r['id']}", "state": r["utt"], "questions": {"massive_niyet": question},
                  "meta": {"source": "AmazonScience/massive tr-TR", "license": "CC BY 4.0"},
                  "labels": {"final": {"massive_niyet": intent(r)}}}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(rows)} örnek, {len(intents)} niyet → {out}")


if __name__ == "__main__":
    main()
