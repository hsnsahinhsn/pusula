<div align="center">

<h1><picture><source media="(prefers-color-scheme: dark)" srcset="site/brand/pusula-logo-reverse.svg"><img src="site/brand/pusula-logo.svg" alt="Pusula" width="300"></picture></h1>

### The first full-featured Turkish decision model

[English](README.md) • [Türkçe](README_TR.md) • [Website](https://hsnsahinhsn.github.io/pusula/en/)

[![CI](https://github.com/hsnsahinhsn/pusula/actions/workflows/ci.yml/badge.svg)](https://github.com/hsnsahinhsn/pusula/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hsnsahinhsn/pusula/blob/main/examples/pusula_nano_colab.ipynb)

</div>

Decision models answer typed questions about a text (for example, which team should handle a ticket, whether it
contains a prompt injection, or how urgent it is) and return a **probability for every option** instead of
generated text. Existing decision models (Decision 3.0, Jev, Laya and others) were developed for English, and their
accuracy on Turkish is limited. **Pusula is Decision 3.0 adapted to Turkish.**

- **Turkish first.** [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (Decision 3.0) has been fine-tuned on Turkish
  decision data from 20 domains, including banking, accounting and tax, law, health, e-commerce, logistics and
  public services. Informal writing, typos, missing Turkish characters and mixed Turkish–English text are covered.
- **Options defined at request time.** Questions and their options are specified in each request.
- **Three question types.** `choice` (one option), `noul` (yes/no) and `score` (ordered levels).
- **Drop-in d3 replacement.** The request and answer format of d3's `/v1/systemone` is preserved.
- **Built-in Turkish taxonomy and templates.** 20 domains, 103 subdomains and 333 intents, together with 20
  ready-made templates (topic and intent, 14 checks for messages and 4 for AI agents). Routing through an
  institution's own topic list is also supported.
- **Turkish decision benchmark.** Pusula has been evaluated against the base Decision 3.0 model on independently
  labeled Turkish test sets; the results are published with the model. The benchmark is available on
  [Hugging Face](https://huggingface.co/datasets/hsnsahinhsn/turkish-decision-benchmark).

> **Status:** v1.0 (18 October 2026). Weights: [Pusula](https://huggingface.co/hsnsahinhsn/pusula) ·
> [Pusula Nano](https://huggingface.co/hsnsahinhsn/pusula-nano). Earlier versions are listed under [Releases](#releases).

## Models

| Model | Size | Runs on | Benchmark / gold set | Use it for |
|---|---|---|---|---|
| **Pusula** (Turkish Decision 3.0) | 26B: a 0.9 GB LoRA adapter on top of `vllm-sr/d3` | GPU: ≥ 64 GB in BF16, ≥ 24 GB in 4-bit | **92.6 / 90.4** | Highest accuracy, for built-in and custom questions |
| **Pusula Nano** | 149M ([TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) cross-encoder) | CPU or any GPU | 82.7 / 84.6 | High volume and low latency on the built-in questions |

Both models accept identical requests.

---

## Results

Accuracy (%). Evaluation labels were assigned blind and independently by two annotators (Claude Opus 5.5 and
Claude Sonnet 5.5); disagreements were adjudicated, and all labels follow the published
[labeling policy](pusula/data/label_policy.md). The real customer messages were labeled blind by Claude under the same
policy. Evaluation labels are used exclusively for evaluation and are never used as training labels.

| Evaluation | Questions | Decision 3.0 | **Pusula** | Pusula Nano |
|---|---|---|---|---|
| Real customer messages (500 held-out, licensed) | 4,500 | 77.4 | **85.6** | 84.7 |
| Long customer complaints (500 held-out, licensed) | 4,500 | 81.4 | **88.0** | 84.6 |
| [Turkish Decision Benchmark](benchmark/README.md) (2,194 messages) | 19,861 | 88.5 | **92.6** | 82.7 |
| Gold test set (759 held-out messages) | 7,015 | 85.5 | **90.4** | 84.6 |
| Your own questions (497 messages × 3 ad-hoc questions) | 1,491 | **94.8** | 94.2 | 87.6 |
| MASSIVE tr-TR, 60-way intent, zero-shot (human-written) | 2,974 | **85.3** | 85.2 | 36.4 |
| OffensEval 2020 Turkish, offensive language (real tweets) | 3,528 | **88.4** | 88.3 | 30.6 |

Gold test set, per question:

| Question | Decision 3.0 | **Pusula** |
|---|---|---|
| Urgency (4 levels) | 53.2 | **78.9** |
| Human handoff | 81.5 | **89.2** |
| Domain | 82.1 | **86.8** |
| Emotion | 85.0 | **88.3** |
| Security risk | 94.2 | **98.0** |
| Action required | 88.9 | **90.5** |
| Personal data | **96.4** | 96.3 |
| Intent | 96.8 | **97.0** |

- **Real customer messages:** Evaluation was performed on two held-out sets of 500 anonymized messages licensed
  from a commercial partner: customer messages (reviews, complaints and questions) and long, detailed customer
  complaints. These sets are not published. Other messages from the same sources were used for training and were
  labeled by an open model (Qwen).
- **Urgency is partly subjective:** the two annotators agree on 84.4% of messages, which bounds the attainable
  score (Pusula: 78.9).
- **Custom questions:** Pusula performs on par with Decision 3.0 (94.2 vs 94.8; custom questions in the gold set
  +1.6, OffensEval −0.1, MASSIVE −0.1).
- **Pusula Nano** was trained mainly on the built-in questions. On real messages its accuracy is close to that of
  Pusula (84.7 vs 85.6). For tasks that differ from the built-in questions, or that have many options, Pusula is
  recommended (OffensEval 88.3 vs 30.6, MASSIVE 85.2 vs 36.4).
- MASSIVE and OffensEval are not included in the training data.

## Contents

- [Installation](#installation)
- [Concepts](#concepts)
- [Python API](#python-api)
- [HTTP API](#http-api)
- [Built-in taxonomy and routing](#built-in-taxonomy-and-routing)
  - [Routing through an institution's own list](#routing-through-an-institutions-own-list)
  - [The 20 templates](#the-20-templates)
- [Writing good questions](#writing-good-questions)
- [Using confidence](#using-confidence)
- [Examples](#examples)
- [Training your own model](#training-your-own-model)
- [Releases](#releases)
- [Contributors and citation](#contributors-and-citation)
- [License and acknowledgements](#license-and-acknowledgements)

---

## Installation

```bash
pip install "pusula[serve,d3] @ git+https://github.com/hsnsahinhsn/pusula"
hf download vllm-sr/d3 --local-dir ~/d3 --exclude "assets/*"     # base Decision 3.0 weights (~52 GB)
hf download hsnsahinhsn/pusula --local-dir ~/pusula                # Pusula adapter (~0.9 GB)
# or the small model, which needs no base weights:
hf download hsnsahinhsn/pusula-nano --local-dir ~/pusula-nano      # Pusula Nano (~0.6 GB)
```

The `serve` extra installs PyTorch, Transformers and FastAPI; the `d3` extra adds PEFT and bitsandbytes for 4-bit
loading. A CUDA build of PyTorch should be installed beforehand ([pytorch.org](https://pytorch.org/get-started/locally/)).
Pusula Nano requires only the `serve` extra.

## Concepts

A request consists of a **state** (the text) and named **questions**:

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

The answer to the request above is shown below (illustrative values):

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

Multiple requests can be processed in a single call; they are packed into as few forward passes as possible:

```python
results = engine.decide_many([(text, questions) for text in messages])
```

Questions may be given as plain dictionaries, as above, or as `pusula.Question` objects.

## HTTP API

The server is started as follows:

```bash
PUSULA_D3_BASE=~/d3 PUSULA_MODEL_DIR=path/to/pusula pusula-serve --port 8080   # Pusula
PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --port 8080                  # Pusula Nano
```

By default, the server listens on `127.0.0.1` only. For access from other machines, an API key must be set and
`--host 0.0.0.0` passed; clients then include `Authorization: Bearer <key>` (or `X-API-Key: <key>`) in every request
to a `/v1/*` path:

```bash
PUSULA_API_KEY=change-me PUSULA_MODEL_DIR=path/to/pusula-nano pusula-serve --host 0.0.0.0 --port 8080
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/systemone` | One `{state, questions}` request (d3-compatible) |
| `POST` | `/v1/systemone/batch` | `{"requests": [...]}`, up to 64 requests |
| `POST` | `/v1/scan` | A long text checked part by part (see [Long texts](#long-texts)) |
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

Invalid questions are rejected with HTTP 422 and the reason. The server is configured through environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PUSULA_D3_BASE` | — | Base `vllm-sr/d3` directory; required for Pusula (`PUSULA_MODEL_DIR` then points to the adapter) |
| `PUSULA_MODEL_DIR` | — | Pusula adapter directory, or a Pusula Nano model directory |
| `PUSULA_DEVICE` | auto | `cuda`, `cuda:1`, `cpu`… |
| `PUSULA_OWN_BASE` | 1 | Pusula only: custom questions (not built-in) are answered by base Decision 3.0 with the adapter disabled; `0` uses the adapter for every question |
| `PUSULA_CORS_ORIGINS` | none | Comma-separated allowed origins for browser clients |
| `PUSULA_MAX_QUESTIONS` | 32 | Questions per request |
| `PUSULA_MAX_BATCH` | 64 | Requests per batch call |
| `PUSULA_MAX_STATE_CHARS` | 60000 | Maximum state length |
| `PUSULA_MAX_SCAN_CHARS` | 200000 | Maximum text length for `/v1/scan` |

### Long texts

`POST /v1/scan` (or `engine.scan(text, questions)`) checks a long text, such as a web page, a document or an e-mail
thread, part by part. The text is split into overlapping parts that the model reads whole; for every question, the
answer of the part with the strongest finding is returned, together with its index in `part`. The request format is
that of `/v1/systemone`, and the part size may be set with `window_chars`. For `choice` questions, the neutral option
should be listed first (as `guvenli` is in `guvenlik`). Texts of up to 200,000 characters are accepted
(`PUSULA_MAX_SCAN_CHARS`).

## Built-in taxonomy and routing

> The 20 templates and routing through an institution's own list, as described in this section, are introduced in
> v2.0 (in development). Models up to v1.0 were trained on the earlier six universal questions.

`POST /v1/route` (or `engine.route(text)`) routes a message in three steps, **alan** (domain), **alt_alan**
(subdomain) and **niyet** (intent), and answers the universal questions in the same call:

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

The universal questions to be answered are selected with `universal`: `true` (the default) asks all 14, `false`
asks none, and a list of names asks only those. Fewer questions give faster answers:

```bash
curl -s localhost:8080/v1/route -H 'Content-Type: application/json' -d '{
  "state": "Kartımdan bilgim dışında 4.500 TL çekilmiş, hemen bloke edin!",
  "universal": ["aciliyet", "dolandiricilik", "insana_devret"]
}'
```

### Routing through an institution's own list

An institution's own topic list may be passed in `topics`; the built-in domains are then not used. Each topic is
given with a description, and intents may be listed under a topic. The topic (`konu`) is decided first; where the
chosen topic lists intents, the intent (`niyet`) is decided next:

```json
{
  "state": "Geçen hafta yaptırdığım EKG'nin sonucunu sistemde göremiyorum.",
  "universal": ["aciliyet"],
  "topics": {
    "kardiyoloji": {
      "description": "Kalp ve damar hastalıkları",
      "intents": {"randevu": "Kardiyoloji randevusu almak veya değiştirmek",
                  "sonuc": "Tetkik veya test sonucunu sormak"}
    },
    "faturalama": "Ödeme, fatura ve sigorta işlemleri",
    "diger": "Yukarıdakilerin hiçbiri"
  }
}
```

The answer contains `"path": "kardiyoloji.sonuc"` together with the answers to `konu`, `niyet` and the selected
universal questions. An invalid list is rejected with HTTP 422. In Python, the same call is
`engine.route(text, topics={...}, universal=["aciliyet"])`.

### Built-in domains

| Domain key | Domain | Subdomains | Intents |
|---|---|---:|---:|
| `finans_bankacilik` | Finans ve Bankacılık | 7 | 28 |
| `muhasebe_vergi` | Muhasebe ve Vergi | 6 | 22 |
| `hukuk_uyum` | Hukuk ve Mevzuat Uyumu | 6 | 22 |
| `saglik` | Sağlık | 6 | 17 |
| `sigorta_sosyal_guvenlik` | Sigorta ve Sosyal Güvenlik | 4 | 14 |
| `ticaret_eticaret` | Ticaret ve E-Ticaret | 6 | 21 |
| `lojistik_tedarik` | Lojistik ve Tedarik Zinciri | 6 | 23 |
| `abonelik_hizmetleri` | Telekom ve Enerji Abonelikleri | 4 | 15 |
| `teknoloji_bt` | Teknoloji ve Bilgi İşlem | 7 | 24 |
| `insan_kaynaklari` | İnsan Kaynakları | 6 | 19 |
| `kamu_hizmetleri` | Kamu Hizmetleri | 5 | 15 |
| `egitim` | Eğitim | 5 | 15 |
| `seyahat_konaklama` | Seyahat ve Konaklama | 5 | 15 |
| `gayrimenkul` | Gayrimenkul | 4 | 12 |
| `otomotiv_mobilite` | Otomotiv ve Mobilite | 5 | 13 |
| `sanayi_uretim_tarim` | Sanayi, Üretim ve Tarım | 5 | 17 |
| `kisisel_verimlilik` | Kişisel Verimlilik ve Ofis | 4 | 12 |
| `medya_eglence` | Medya ve Eğlence | 5 | 12 |
| `kriz_destek` | Kriz ve Psikososyal Destek | 3 | 9 |
| `genel_sohbet` | Genel ve Kapsam Dışı | 4 | 8 |

Every domain except `kisisel_verimlilik` and `kriz_destek` contains a `geri_bildirim` subdomain (complaints and
praise about that domain's services that contain no request), so that a review is routed to the domain it is about.

### The 20 templates

Ready-made questions are provided for the most common decisions. A template is an ordinary question that has been
written in advance; templates and custom questions may be combined in a single request.

| # | Name | Type | Question | Options |
|---:|---|---|---|---|
| 1 | `alan` / `konu` | choice | What is the topic? | The 20 built-in domains, or the institution's own list |
| 2 | `niyet` | choice | What is the intent? | The intents of the chosen domain or topic; the subdomain is decided as an intermediate step |
| 3 | `mesaj_turu` | choice | What type of message is it? | `soru`, `talep`, `sikayet`, `ovgu`, `oneri`, `bilgi_verme`, `spam_anlamsiz` |
| 4 | `aciliyet` | score | How quickly should it be handled? | Rutin · Yakın · Yüksek · Kritik |
| 5 | `duygu` | choice | What is the writer's dominant emotion? | `notr`, `memnun`, `endiseli`, `uzgun`, `hayal_kirikligi`, `ofkeli` |
| 6 | `eksik_bilgi` | noul | Is a critical detail missing or contradictory, so that a question must be asked first? | |
| 7 | `eylem_gerekli` | noul | Is a concrete action (ticket, cancellation, refund, block, tool call) required beyond a reply? | |
| 8 | `insana_devret` | noul | Should the case be handed over to a human expert? | |
| 9 | `kisisel_veri` | noul | Does the text contain personal or sensitive data? | |
| 10 | `guvenlik` | choice | Is there an attack on, or a manipulation attempt against, the AI? | `guvenli`, `prompt_enjeksiyonu`, `jailbreak_rol_yapma`, `zararli_veya_yasadisi`, `yetkisiz_islem_talebi` |
| 11 | `hakaret` | choice | Does the text contain insults, swearing or hate speech? | `yok`, `hakaret_kufur`, `nefret_soylemi` |
| 12 | `dolandiricilik` | noul | Is fraud suspected (phishing, a fake document, impersonation, or a report of being defrauded)? | |
| 13 | `can_guvenligi` | choice | Is anyone's life or safety at risk? | `yok`, `kendine_zarar`, `siddet_tehdit`, `tibbi_acil` |
| 14 | `yasal_risk` | noul | Does the text carry a legal risk for the institution? | |
| 15 | `ayrilma_riski` | choice | Is the writer inclined to leave or cancel? | `hayir`, `risk_var`, `acikca_istiyor` |
| 16 | `satis_firsati` | noul | Is there a sales opportunity? | |
| 17 | `islem_riski` | score | How much harm could the agent's proposed action cause? | Düşük · Orta · Yüksek · Kritik |
| 18 | `veri_disari` | noul | Does the proposed action send confidential data outside the institution? | |
| 19 | `kaynaga_dayanma` | choice | Is the agent's answer grounded in the given source? | `dayaniyor`, `kismen`, `uydurma` |
| 20 | `gorev_tamamlandi` | choice | Does the agent's result fulfil the request? | `tamamlandi`, `kismen`, `tamamlanmadi` |

Templates 1 and 2 are the routing steps above, and templates 3 to 16 are the universal questions of `/v1/route`.
Templates 17 to 20 are asked about the working context of an AI agent. Such a state is written in labeled sections,
as needed: `Kullanıcı isteği:` (the user's request), `Önerilen işlem:` (the proposed action), `Kaynak:` (the
source) and `Ajanın cevabı:` (the agent's answer). Whether an action needs approval is derived from `islem_riski`
(for example, from `Yüksek` upward), and the need for clarification is given by `eksik_bilgi`.

```python
from pusula import load_taxonomy
tax = load_taxonomy()
state = ("Kullanıcı isteği: İade süresi kaç gün?\n"
         "Kaynak: Ürünler teslim tarihinden itibaren 14 gün içinde iade edilebilir.\n"
         "Ajanın cevabı: Ürünü 30 gün içinde iade edebilirsiniz.")
answers = engine.decide(state, {"kaynaga_dayanma": tax.agent["kaynaga_dayanma"],
                                "gorev_tamamlandi": tax.agent["gorev_tamamlandi"]})
```

The complete taxonomy, including every subdomain and intent, is provided in
[`pusula/data/taxonomy.yaml`](pusula/data/taxonomy.yaml) and at `GET /v1/taxonomy`; the boundary cases of every
template are described in the [labeling policy](pusula/data/label_policy.md). Any of these questions can also be
reused in custom requests:

```python
questions = {"alan": tax.domain_question(), "aciliyet": tax.universal["aciliyet"], "my_question": {...}}
```

## Writing good questions

- **Every option should be described.** The model reads the descriptions; `"iade": "Teslim alınan ürünü geri
  göndermek"` performs considerably better than a bare key.
- **Options should be mutually exclusive.** Where two options overlap, the boundary should be stated
  (`"Kapsamaz: ..."`, i.e. "does not cover").
- **Instructions and options should be written in Turkish,** the language the model was trained on.
- **Up to about 20 options per question are recommended.** Larger label sets are best handled hierarchically
  (category first, then the options within it), as `/v1/route` does.
- **The appropriate type should be used.** Yes/no questions use `noul`, ordered levels use `score` (lowest first),
  and all other questions use `choice`.
- **A catch-all option** (`"diger": "Yukarıdakilerin hiçbiri"`) should be added when inputs may fall outside the
  listed options.

## Using confidence

- `choice` / `score`: `confidence` is the probability of the selected option.
- `noul`: the answer itself is a probability, and the threshold is chosen by the application (for example `> 0.5`,
  or `> 0.8` for high-risk actions).
- A common pattern is to act automatically above a high threshold, to request confirmation in the middle band and
  to hand over to a human below it. Thresholds should be calibrated on a sample of the application's own traffic.

## Examples

- [`examples/agentic_router.py`](examples/agentic_router.py): guardrail check, tool selection, approval and urgency
  for an agent, obtained in a single call.
- [`examples/pusula_nano_colab.ipynb`](examples/pusula_nano_colab.ipynb): Pusula Nano on Google Colab (custom
  questions, built-in routing and batch processing), which can be opened without installation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hsnsahinhsn/pusula/blob/main/examples/pusula_nano_colab.ipynb)

## Training your own model

All components used to build Pusula are included in this repository, so Decision 3.0 can be fine-tuned on data
from other domains or languages. The pipeline generates labeled decision data with a local LLM, cleans it and
fine-tunes d3 with QLoRA:

```bash
pip install -e ".[train,generate,d3]"
bash scripts/pipeline.sh my-run
# generate → judge → fetch_d3 → teach → build → curate → gold → d3train → evaluate
```

Each stage can be resumed. Options are documented in the module docstrings in [`pusula/`](pusula/).

The judgment questions (urgency, human handoff, action required) follow a single written
[labeling policy](pusula/data/label_policy.md), which can be adapted to a specific product. After a policy change,
`python -m pusula.judge --policy --refresh --questions …` relabels these questions with the open-weight judge, and
`python -m pusula.relabel` measures the new labels against the gold set and replaces them in the training data.
`python -m pusula.annotate` (`pip install -e ".[annotate]"`, Claude API) produces blind gold and benchmark labels with
two annotators and adjudication. Such labels should be used for evaluation only, since the terms of most commercial
models do not permit training competing models on their outputs.

## Releases

Each release is tagged in this repository and in both Hugging Face repositories. Earlier weights are obtained with
`hf download hsnsahinhsn/pusula --revision <tag>`.

| Version | Date | Real customer messages (Pusula / Nano) | Notes |
|---|---|---|---|
| `v1.0` | 18 October 2026 | 85.6 / 84.7 | Current release. Additional licensed training data (long customer complaints) and a second held-out real evaluation set. |
| `v1.0-onizleme` | 17 October 2026 | 84.1 / 82.9 | Preview release. |

## Contributors and citation

Pusula is developed by the following contributors:

| Name | Role | GitHub | Hugging Face |
|---|---|---|---|
| Hasan Şahin | Project lead | [hsnsahinhsn](https://github.com/hsnsahinhsn) | [hsnsahinhsn](https://huggingface.co/hsnsahinhsn) |
| Ahmet Sayman | Contributor | [asayman](https://github.com/asayman) | [asaymantr](https://huggingface.co/asaymantr) |

If Pusula is used in research or in a product, the following citation is requested. The same information is provided
in [`CITATION.cff`](CITATION.cff) and through the "Cite this repository" button on GitHub.

```bibtex
@software{pusula2026,
  title   = {Pusula: The First Full-Featured Turkish Decision Model},
  author  = {Şahin, Hasan and Sayman, Ahmet},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/hsnsahinhsn/pusula},
  license = {Apache-2.0}
}
```

## License and acknowledgements

Pusula is released under the Apache 2.0 license and builds on
[TabiBERT](https://huggingface.co/boun-tabilab/TabiBERT) (Boğaziçi University TABILAB),
[vllm-sr/d3](https://huggingface.co/vllm-sr/d3) (vLLM Semantic Router) and
[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B), all licensed under Apache 2.0. The d3 prompt contract and the
non-causal attention hook in `pusula/d3model.py` follow vllm-sr/d3 (where they were adapted from
perplexity-ai/pplx-decider-v1.1-27b, Apache 2.0).
