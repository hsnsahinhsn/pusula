<div align="center">

# Pusula

### The first full-featured Turkish decision model

[English](README.md) • [Türkçe](README_TR.md)

[![CI](https://github.com/hsnsahinhsn/pusula/actions/workflows/ci.yml/badge.svg)](https://github.com/hsnsahinhsn/pusula/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

</div>

Decision models answer typed questions about a text — route this ticket, is this a prompt injection,
how urgent is it — and return a **probability for every option** instead of generating text. Today's
decision models (Decision 3.0, Jev, Laya, ...) are built for English and stumble on Turkish.
**Pusula is Decision 3.0 trained for Turkish.**

- **Turkish first.** [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (Decision 3.0) fine-tuned on Turkish
  decision data across 20 domains: banking, accounting and tax, law, health, e-commerce, logistics, public
  services and more — informal writing, typos, missing Turkish characters and mixed Turkish–English included.
- **Any options, at request time.** You define the question and its options in each request.
- **Three question types.** `choice` (pick one), `noul` (yes/no), `score` (ordered levels).
- **Drop-in d3 replacement.** Same request and answer format as d3's `/v1/systemone`.
- **Built-in Turkish taxonomy.** 20 domains → 86 subdomains → 299 intents, plus six universal checks
  (urgency, action needed, human handoff, security risk, emotion, personal data).
- **Turkish decision benchmark.** Pusula is evaluated against base Decision 3.0 on an independently
  labeled Turkish test set; results are published with the model.

> **Status:** v1.0. Weights: [Pusula](https://huggingface.co/hsnsahinhsn/pusula) ·
> [Pusula Mini](https://huggingface.co/hsnsahinhsn/pusula-mini) · [Pusula Nano](https://huggingface.co/hsnsahinhsn/pusula-nano).

## Models

| Model | Size | Runs on | Benchmark / gold set | Use it for |
|---|---|---|---|---|
| **Pusula** (Turkish Decision 3.0) | 26B: a 0.9 GB LoRA adapter on top of `vllm-sr/d3` | GPU: ≥ 64 GB in BF16, ≥ 24 GB in 4-bit | **92.8 / 90.3** | The highest accuracy, for the built-in questions and your own |
| **Pusula Mini** | 4.7B (Qwen3.5-4B, fully fine-tuned, d3 format) | GPU (weights: 9.1 GB in BF16) | 80.9 / 84.5 | A smaller GPU; the strongest small model on tasks unlike its training data |
| **Pusula Nano** | 149M ([TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) cross-encoder) | CPU or any GPU | 81.8 / 84.2 | High volume and low latency on the built-in questions |

All of them answer exactly the same requests.

---

## Results

Accuracy (%). The evaluation labels come from two annotators working blind and independently (Claude Opus 5.5 and
Claude Sonnet 5.5), with disagreements adjudicated, all following the published
[labeling policy](pusula/data/label_policy.md). They are used only for evaluation, never as training labels.

| Evaluation | Questions | Decision 3.0 | **Pusula** | Pusula Mini | Pusula Nano |
|---|---|---|---|---|---|
| [Turkish Decision Benchmark](benchmark/README.md) (2,194 messages) | 19,861 | 88.5 | **92.8** | 80.9 | 81.8 |
| Gold test set (759 held-out messages) | 7,015 | 85.5 | **90.3** | 84.5 | 84.2 |
| Your own questions (497 messages × 3 ad-hoc questions) | 1,491 | **94.8** | 93.1 | 85.0 | 85.6 |
| MASSIVE tr-TR, 60-way intent, zero-shot (human-written) | 2,974 | **85.3** | 84.7 | 33.3 | 30.9 |
| OffensEval 2020 Turkish, offensive language (real tweets) | 3,528 | 88.4 | **88.8** | 76.9 | 27.5 |

Gold test set, per question:

| Question | Decision 3.0 | **Pusula** |
|---|---|---|
| Urgency (4 levels) | 53.2 | **80.2** |
| Human handoff | 81.5 | **89.1** |
| Domain | 82.1 | **87.4** |
| Emotion | 85.0 | **88.9** |
| Security risk | 94.2 | **97.8** |
| Action required | 88.9 | **89.2** |
| Personal data | **96.4** | 96.0 |
| Intent | **96.8** | 95.1 |

- **Urgency is partly subjective:** the two annotators agree on 84.4% of messages, so 80.2 is close to the ceiling.
- **Your own questions:** Pusula is on par with Decision 3.0 overall (custom questions in the gold set +2.2, OffensEval
  +0.4, MASSIVE −0.6) but behind on questions with 10 or more options (86.7 vs 90.1). An update for long option lists
  is planned.
- **Mini and Nano** are trained mainly on the built-in questions. For tasks unlike them, with many options, use Pusula;
  among the small models, Mini handles such tasks far better (OffensEval 76.9 vs 27.5).
- MASSIVE and OffensEval are not part of the training data.

## Contents

- [Installation](#installation)
- [Concepts](#concepts)
- [Python API](#python-api)
- [HTTP API](#http-api)
- [Built-in taxonomy and routing](#built-in-taxonomy-and-routing)
- [Writing good questions](#writing-good-questions)
- [Using confidence](#using-confidence)
- [Examples](#examples)
- [Training your own model](#training-your-own-model)
- [License and acknowledgements](#license-and-acknowledgements)

---

## Installation

```bash
pip install "pusula[serve,d3] @ git+https://github.com/hsnsahinhsn/pusula"
hf download vllm-sr/d3 --local-dir ~/d3 --exclude "assets/*"     # base Decision 3.0 weights (~52 GB)
hf download hsnsahinhsn/pusula --local-dir ~/pusula                # Pusula adapter (~0.9 GB)
# or one of the small models, which need no base weights:
hf download hsnsahinhsn/pusula-mini --local-dir ~/pusula-mini      # Pusula Mini (~9.1 GB)
hf download hsnsahinhsn/pusula-nano --local-dir ~/pusula-nano      # Pusula Nano (~0.6 GB)
```

The `serve` extra installs PyTorch, Transformers and FastAPI; `d3` adds PEFT and bitsandbytes for the
4-bit model. Install a CUDA build of PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)).
Pusula Mini and Pusula Nano need only the `serve` extra.

## Concepts

A request has a **state** and named **questions**:

```json
{
  "state": "Kartımdan bilgim dışında 4.500 TL çekilmiş, hemen bloke edin!",
  "questions": {
    "ekip": {
      "type": "choice",
      "instructions": "Bu talebi hangi ekip ele almalı?",
      "criteria": {
        "dolandiricilik": "Şüpheli işlem ve dolandırıcılık",
        "kart_islemleri": "Kart başvurusu, limit ve şifre",
        "genel_destek": "Diğer sorular"
      }
    },
    "acil": { "type": "noul", "instructions": "Hemen müdahale gerekiyor mu?" },
    "risk": {
      "type": "score",
      "instructions": "Finansal risk ne düzeyde?",
      "criteria": ["Düşük", "Orta", "Yüksek"]
    }
  }
}
```

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | Object mapping option keys to descriptions (`null` allowed) | `choice`, `probabilities`, `confidence` |
| `noul` | Optional; may describe `"false"` and `"true"` | `noul` = probability of *true* |
| `score` | Ordered list of 2–10 levels, lowest first | `score` = expected level (0…n-1), `probabilities`, `confidence`, `legend` |

Answer for the request above (illustrative values):

```json
{
  "answers": {
    "ekip": {
      "type": "choice",
      "choice": "dolandiricilik",
      "probabilities": {"dolandiricilik": 0.94, "kart_islemleri": 0.05, "genel_destek": 0.01},
      "confidence": 0.94
    },
    "acil": {"type": "noul", "noul": 0.97},
    "risk": {
      "type": "score",
      "score": 1.86,
      "probabilities": {"0": 0.02, "1": 0.10, "2": 0.88},
      "confidence": 0.88,
      "legend": {"0": "Düşük", "1": "Orta", "2": "Yüksek"}
    }
  },
  "model": "pusula",
  "latency_ms": 9.8
}
```

## Python API

```python
from pusula.inference import D3Engine, DecisionEngine

engine = D3Engine.load("~/d3", adapter="~/pusula")      # Pusula (Turkish Decision 3.0)
# engine = D3Engine.load("~/pusula-mini")                # Pusula Mini (d3 format, no adapter)
# engine = DecisionEngine.load("~/pusula-nano")          # Pusula Nano (GPU or CPU)

answers = engine.decide(
    "Siparişim 5 gündür kargoda, hâlâ gelmedi. Artık iptal etmek istiyorum.",
    {
        "niyet": {
            "type": "choice",
            "instructions": "Müşteri ne istiyor?",
            "criteria": {
                "kargo_takip": "Gönderinin nerede olduğunu öğrenmek",
                "iptal": "Siparişi iptal etmek",
                "iade": "Teslim alınan ürünü iade etmek",
            },
        },
        "temsilci": {"type": "noul", "instructions": "Bir insan temsilciye aktarılmalı mı?"},
    },
)
print(answers["niyet"]["choice"], answers["niyet"]["confidence"])
print(answers["temsilci"]["noul"])
```

Many requests at once (packed into as few forward passes as possible):

```python
results = engine.decide_many([(text, questions) for text in messages])
```

Questions can be plain dicts as above or `pusula.Question` objects.

## HTTP API

Start the server:

```bash
PUSULA_D3_BASE=~/d3 PUSULA_MODEL_DIR=path/to/pusula pusula-serve --port 8080   # Pusula
PUSULA_D3_BASE=~/pusula-mini pusula-serve --port 8080                         # Pusula Mini
PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --port 8080                  # Pusula Nano
```

The server listens on `127.0.0.1` by default. To reach it from other machines, set an API key first and pass
`--host 0.0.0.0`; clients then send `Authorization: Bearer <key>` (or `X-API-Key: <key>`) to every `/v1/*` path:

```bash
PUSULA_API_KEY=change-me PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --host 0.0.0.0 --port 8080
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/systemone` | One `{state, questions}` request (d3-compatible) |
| `POST` | `/v1/systemone/batch` | `{"requests": [...]}`, up to 64 requests |
| `POST` | `/v1/route` | Route a message through the built-in taxonomy |
| `GET` | `/v1/taxonomy` | The built-in taxonomy as JSON |
| `GET` | `/health` | Liveness; always 200 |
| `GET` | `/ready` | 200 once the model is loaded, otherwise 503 |

```bash
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Faturama kullanmadığım bir paket için 450 TL yansıtılmış.",
  "questions": {
    "departman": {"type": "choice", "instructions": "Hangi departman?",
                  "criteria": {"fatura": "Fatura ve ödeme", "teknik": "Arıza ve teknik destek"}},
    "iade": {"type": "noul", "instructions": "Müşteri para iadesi istiyor mu?"}
  }
}'
```

Invalid questions return HTTP 422 with the reason. Configuration (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `PUSULA_D3_BASE` | — | Base `vllm-sr/d3` directory; set it to serve Pusula (then `PUSULA_MODEL_DIR` is the adapter) |
| `PUSULA_MODEL_DIR` | — | Pusula adapter directory, or a Pusula Nano model directory |
| `PUSULA_DEVICE` | auto | `cuda`, `cuda:1`, `cpu`… |
| `PUSULA_CORS_ORIGINS` | none | Comma-separated allowed origins for browser clients |
| `PUSULA_MAX_QUESTIONS` | 32 | Questions per request |
| `PUSULA_MAX_BATCH` | 64 | Requests per batch call |
| `PUSULA_MAX_STATE_CHARS` | 60000 | Maximum state length |

## Built-in taxonomy and routing

`POST /v1/route` (or `engine.route(text)`) routes a message in three steps — **alan** (domain) →
**alt_alan** (subdomain) → **niyet** (intent) — and also answers the universal questions:

```json
{
  "path": "finans_bankacilik.itiraz_ve_dolandiricilik.supheli_islem",
  "answers": {
    "alan": {"choice": "finans_bankacilik", "confidence": 0.97, "...": "..."},
    "aciliyet": {"type": "score", "score": 2.8, "...": "..."},
    "insana_devret": {"type": "noul", "noul": 0.71}
  }
}
```

| Domain key | Domain | Subdomains | Intents |
|---|---|---:|---:|
| `finans_bankacilik` | Finans ve Bankacılık | 6 | 26 |
| `muhasebe_vergi` | Muhasebe ve Vergi | 5 | 20 |
| `hukuk_uyum` | Hukuk ve Mevzuat Uyumu | 5 | 20 |
| `saglik` | Sağlık | 5 | 15 |
| `sigorta_sosyal_guvenlik` | Sigorta ve Sosyal Güvenlik | 3 | 12 |
| `ticaret_eticaret` | Ticaret ve E-Ticaret | 5 | 19 |
| `lojistik_tedarik` | Lojistik ve Tedarik Zinciri | 5 | 21 |
| `abonelik_hizmetleri` | Telekom ve Enerji Abonelikleri | 3 | 13 |
| `teknoloji_bt` | Teknoloji ve Bilgi İşlem | 6 | 22 |
| `insan_kaynaklari` | İnsan Kaynakları | 5 | 17 |
| `kamu_hizmetleri` | Kamu Hizmetleri | 4 | 13 |
| `egitim` | Eğitim | 4 | 13 |
| `seyahat_konaklama` | Seyahat ve Konaklama | 4 | 13 |
| `gayrimenkul` | Gayrimenkul | 3 | 10 |
| `otomotiv_mobilite` | Otomotiv ve Mobilite | 4 | 11 |
| `sanayi_uretim_tarim` | Sanayi, Üretim ve Tarım | 4 | 15 |
| `kisisel_verimlilik` | Kişisel Verimlilik ve Ofis | 4 | 12 |
| `medya_eglence` | Medya ve Eğlence | 4 | 10 |
| `kriz_destek` | Kriz ve Psikososyal Destek | 3 | 9 |
| `genel_sohbet` | Genel ve Kapsam Dışı | 4 | 8 |

Universal questions (asked of every message):

| Name | Type | Options |
|---|---|---|
| `aciliyet` | score | Rutin · Yakın · Yüksek · Kritik |
| `eylem_gerekli` | noul | Does it need a concrete action (ticket, cancel, refund, block, tool call)? |
| `insana_devret` | noul | Should a human expert take over? |
| `guvenlik` | choice | `guvenli`, `prompt_enjeksiyonu`, `jailbreak_rol_yapma`, `zararli_veya_yasadisi`, `yetkisiz_islem_talebi`, `spam_anlamsiz` |
| `duygu` | choice | `notr`, `memnun`, `endiseli`, `uzgun`, `hayal_kirikligi`, `ofkeli`, `caresiz_kriz` |
| `kisisel_veri` | noul | Does it contain personal or sensitive data? |

The full taxonomy with every subdomain and intent is in
[`pusula/data/taxonomy.yaml`](pusula/data/taxonomy.yaml) and at `GET /v1/taxonomy`. You can also
reuse any of these questions in your own requests:

```python
from pusula import load_taxonomy
tax = load_taxonomy()
questions = {"alan": tax.domain_question(), **tax.universal, "my_question": {...}}
```

## Writing good questions

- **Describe every option.** The model reads the description; `"iade": "Teslim alınan ürünü geri göndermek"`
  works much better than a bare key.
- **Keep options mutually exclusive.** When two options overlap, say where the line is
  (`"Kapsamaz: ..."` / "does not cover").
- **Write instructions and options in Turkish.** That is what the model is trained on.
- **Prefer up to ~20 options per question.** For larger label sets, ask hierarchically
  (category first, then the options inside it), like `/v1/route` does.
- **Use the right type.** Yes/no → `noul`; ordered levels → `score` (lowest first); everything else → `choice`.
- **Add a catch-all** (`"diger": "Yukarıdakilerin hiçbiri"`) when the input may fall outside your options.

## Using confidence

- `choice` / `score`: `confidence` is the probability of the chosen option.
- `noul`: the answer itself is a probability; pick your own threshold (e.g. `> 0.5`, or `> 0.8` for risky actions).
- A typical pattern: act automatically above a high threshold, ask for confirmation in the middle band,
  hand over to a human below it. Tune the thresholds on a sample of your own traffic.

## Examples

- [`examples/agentic_router.py`](examples/agentic_router.py) — guardrail check, tool selection, approval
  and urgency for an agent, in a single call.

## Training your own model

Everything used to build Pusula is in this repository, so you can fine-tune Decision 3.0 on your own
domain or language data. The pipeline writes labeled decision data with a local LLM, cleans it, and
fine-tunes d3 with QLoRA:

```bash
pip install -e ".[train,generate,d3]"
bash scripts/pipeline.sh my-run
# generate → judge → fetch_d3 → teach → build → curate → gold → d3train → evaluate
```

Each stage is resumable. See the module docstrings in [`pusula/`](pusula/) for options.

The judgment questions (urgency, human handoff, action needed) follow one written
[labeling policy](pusula/data/label_policy.md); edit it to match your own product. After a policy change,
`python -m pusula.judge --policy --refresh --questions …` relabels those questions with the open-weight judge and
`python -m pusula.relabel` measures the new labels against your gold set and swaps them into the training data.
`python -m pusula.annotate` (`pip install -e ".[annotate]"`, Claude API) builds blind gold and benchmark labels with
two annotators and adjudication. Use such labels only to evaluate: most commercial model terms do not allow training
competing models on their outputs.

## License and acknowledgements

Apache 2.0. Pusula builds on
[TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) (Boğaziçi University TABILAB),
[vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (vLLM Semantic Router) and
[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B), all Apache 2.0. The d3 prompt contract and
non-causal attention hook in `pusula/d3model.py` follow vllm-sr/d3 (adapted there from
perplexity-ai/pplx-decider-v1.1-27b), Apache 2.0.
