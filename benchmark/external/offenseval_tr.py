"""OffensEval 2020 Turkish (Çöltekin, LREC 2020; CC BY 2.0) as a Pusula evaluation set: real Turkish tweets
(user mentions anonymized), one yes/no question: is the message offensive?

    python benchmark/external/offenseval_tr.py --out runs/external/offenseval_tr_test.jsonl

Source: https://huggingface.co/datasets/coltekin/offenseval2020_tr — please cite Çöltekin (2020), "A Corpus of
Turkish Offensive Language on Social Media". The data is downloaded at run time and not redistributed with Pusula.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "coltekin/offenseval2020_tr"
PARQUET = "refs/convert/parquet"
QUESTION = {"type": "noul",
            "instructions": "Bu mesaj hakaret, küfür, aşağılama veya saldırgan dil içeriyor mu?",
            "criteria": {"false": "Hayır: saldırgan dil yok", "true": "Evet: hakaret, küfür veya saldırgan dil var"}}


def main(argv=None) -> None:
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    table = pq.read_table(hf_hub_download(REPO, f"offenseval2020-turkish/{a.split}/0000.parquet", repo_type="dataset",
                                          revision=PARQUET))
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    label_col = next(c for c, f in features.items() if f.get("_type") == "ClassLabel")
    names = features[label_col]["names"]
    text_col = next(c for c, f in features.items() if f.get("dtype") == "string" and c not in ("id",))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(table.to_pylist()):
            offensive = names[r[label_col]].upper().startswith("OFF")
            ex = {"id": f"offenseval-tr-{r.get('id', i)}", "state": r[text_col],
                  "questions": {"saldirgan_dil": QUESTION},
                  "meta": {"source": REPO, "license": "CC BY 2.0"},
                  "labels": {"final": {"saldirgan_dil": "true" if offensive else "false"}}}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} örnek ({label_col}: {names}, metin: {text_col}) → {out}")


if __name__ == "__main__":
    main()
