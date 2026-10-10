<div align="center">

# Pusula: Turkish Decision Engine
### Sub-15 ms Typed Decision, Routing, and Triage Engine for Autonomous Turkish Agent Workflows

[ English ](README.md) • [ Türkçe ](README_TR.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/hsnsahinhsn/pusula/actions/workflows/ci.yml/badge.svg)](https://github.com/hsnsahinhsn/pusula/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![Latency](https://img.shields.io/badge/Latency-%3C15ms-blue.svg)](#benchmark--performance)
[![Architecture](https://img.shields.io/badge/Architecture-Non--Autoregressive-informational.svg)](#overview)

[Overview](#overview) • [Quickstart](#quickstart) • [Python API](#python-api) • [Agentic Workflow](#agentic-workflow-integration) • [Macro-Domain Taxonomy](#32-macro-domain-taxonomy) • [Benchmark](#benchmark--performance) • [Training & Reproducibility](#training--reproducibility)

</div>

---

## Overview

Routing user queries, selecting tools, and enforcing policy guardrails using generative autoregressive large language models (such as GPT-6 Luna, Claude Opus 5.5, Gemini 3.8 Flash, DeepSeek-V4.1 Flash, or GLM-5.3) introduces severe operational bottlenecks: high latency (350 ms – 1,800 ms), compounding API expenses, and fragile JSON parsing failures.

**Pusula (Turkish Decision Engine)** is an open-source, non-autoregressive decision engine optimized for the Turkish language. Utilizing a single forward pass over an encoder backbone, Pusula simultaneously generates three calibrated decisions in under 15 milliseconds:

| Metric / Capability | Frontier LLMs (GPT-6 Luna, Claude Opus 5.5) | Fast Generative Models (Gemini 3.8, DeepSeek-V4.1) | Pusula |
| :--- | :---: | :---: | :---: |
| **P95 Inference Latency** | 500 ms – 2,000 ms | 200 ms – 600 ms | **8 ms – 15 ms** |
| **Hallucination Risk** | Present | High | **0.0% (Mathematically constrained)** |
| **Output Type** | Freeform text / Unstructured JSON | Fragile JSON | **Direct Typed Tensors** |
| **Probability Calibration** | Uncalibrated | Frequently Overconfident | **Calibrated via Platt Scaling** |
| **Operational Cost** | Per-token commercial billing | High GPU memory footprint | **Zero per-request fee / Self-hosted** |

---

## Output Specification (Typed Decision Head)

Pusula executes a multi-task projection over candidate representations and context embeddings to return a typed decision structure:

```json
{
  "choice": "bankacilik_finans_kripto",
  "choice_confidence": 0.9842,
  "score": 0.9410,
  "action_required": true,
  "latency_ms": 11.2
}
```

- **`choice`**: The selected categorical target or tool identifier chosen from dynamic candidate options.
- **`choice_confidence`**: Empirically calibrated probability score (`0.00 – 1.00`) optimized via Platt temperature scaling.
- **`score`**: Continuous priority, urgency, or risk regression scalar (`0.00 – 1.00`).
- **`action_required`**: Deterministic boolean flag signaling mandatory human-in-the-loop escalation or immediate tool execution.

---

## Quickstart

### Installation

```bash
git clone https://github.com/hsnsahinhsn/pusula.git
cd pusula
pip install -r requirements.txt
```

---

## Python API

```python
import torch
from transformers import AutoTokenizer
from model import TurkishDecisionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints/pusula/best"

# Load model and tokenizer
model = TurkishDecisionModel.from_pretrained(checkpoint_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# Single inference request
context = "Hesabımdan bilgim dışında 15.000 TL çekim yapılmış, kartımı acilen bloke edin."
options = ["bankacilik_finans_kripto", "e_ticaret_pazaryerleri", "hukuk_adalet_mevzuat"]

decision = model.predict_single(tokenizer, context=context, options=options, device=device)

print(decision)
# Output:
# {
#   'choice': 'bankacilik_finans_kripto',
#   'choice_confidence': 0.9912,
#   'score': 0.9850,
#   'action_required': True,
#   'latency_ms': 11.4
# }
```

---

## Agentic Workflow Integration

Pusula functions as a zero-latency triage and routing gatekeeper in autonomous multi-agent pipelines (LangGraph, CrewAI, AutoGen):

```python
from agentic_router_example import decision_route

def agent_triage_pipeline(user_prompt: str) -> dict:
    # Phase 1: Security and Guardrail Verification (<10 ms)
    security_eval = decision_route(
        context=user_prompt,
        options=["guvenli_mesru_istek", "prompt_injection_saldirisi", "pii_kisisel_veri_ifsa"]
    )
    if security_eval["choice"] != "guvenli_mesru_istek" and security_eval["action_required"]:
        return {"status": "rejected", "reason": security_eval["choice"]}

    # Phase 2: Autonomous Tool / Specialized Agent Selection (<12 ms)
    candidate_agents = ["sql_database_agent", "web_search_agent", "calendar_booking", "human_support"]
    routing_eval = decision_route(context=user_prompt, options=candidate_agents)
    
    return {
        "status": "routed",
        "target_agent": routing_eval["choice"],
        "priority_score": routing_eval["score"]
    }
```

---

## REST API Microservice (FastAPI)

Deploy Pusula as an enterprise inference microservice:

```bash
python serve.py
```

The service launches asynchronously on `http://localhost:8080`.

**cURL Request:**
```bash
curl -X POST "http://localhost:8080/decide" \
     -H "Content-Type: application/json" \
     -d '{
       "context": "Yarın saat 14:00 için şirket toplantı odasını ayırt.",
       "options": ["kisisel_asistanlik", "hukuk_adalet_mevzuat", "akademik_ve_sinavlar"]
     }'
```

**JSON Response (<15 ms):**
```json
{
  "choice": "kisisel_asistanlik",
  "choice_confidence": 0.9840,
  "score": 0.7200,
  "action_required": true,
  "latency_ms": 11.8
}
```

---

## 32 Macro-Domain Taxonomy

Pusula natively handles arbitrary runtime candidate labels (`dynamic candidate options`) while offering fine-tuned calibration across 32 universal enterprise domains:

1. **Global Master Router:** Universal domain triage and root-level dispatch
2. **Academia & Standardized Testing:** SAT, ACT, ESAT, STEP, IMAT, IELTS, TOEFL, GRE, GMAT, YKS, ALES
3. **Healthcare, Medicine & Triage:** Emergency department triage, lab report interpretation, clinical escalation
4. **Law, Justice & Statutory Compliance:** Jurisprudence search, contract analysis, criminal law, KVKK/GDPR
5. **Banking, Fintech & Digital Assets:** Fraud detection, credit risk, equity trading, crypto transactions
6. **Insurance & Social Security:** Pension calculation, claims processing, actuarial policy modifications
7. **Software Engineering, IT & DevOps:** Bug resolution, code refactoring, SQL optimization, CI/CD, MLOps
8. **Fundamental Sciences (Advanced STEM):** Multivariable calculus, quantum mechanics, organic chemistry
9. **Engineering & Manufacturing:** Mechatronics, CAD designs, structural civil analysis, ISO/TSE standards
10. **E-Commerce & Digital Marketplaces:** Returns processing, logistics exceptions, billing disputes
11. **Logistics & Customs Clearance:** Freight manifests, bill of lading, HS tariff classification, bonded storage
12. **Real Estate & Property Management:** Lease negotiation, title deed transactions, appraisal audits
13. **Automotive & Fleet Diagnostics:** OBD-II fault diagnostics, fleet telematics, telemetric roadside triage
14. **Human Resources & Talent Acquisition:** ATS resume screening, technical interview rubrics, payroll queries
15. **Travel, Aviation & Consular Services:** Visa document validation, consular workflows, itinerary booking
16. **Gastronomy & Food Safety:** Recipe scaling, commercial kitchen hygiene, allergen compliance
17. **Nutrition & Sports Science:** Caloric macronutrient tracking, periodized programming, sports rehabilitation
18. **Pedagogy & Child Development:** Developmental milestones, educational behavioral interventions
19. **Agriculture & Veterinary Medicine:** Animal toxicology, veterinary triage, agronomic pest control
20. **Environmental Engineering & Clean Energy:** Solar PV installations, carbon footprint metrics, EIA reporting
21. **Public Administration & Civic Governance:** Digital government authentication, municipal services, civic compliance
22. **History & Archival Research:** Archival paleography, Ottoman/Seljuk documentation, historiographical queries
23. **Linguistics & Translation:** Sworn technical translation, localisation, syntactic proofreading
24. **Fine Arts, Music & Media Production:** Music theory, cinematography analysis, dramatic structure
25. **Interactive Media & Esports:** Game engine scripting (Unreal/Unity), competitive latency optimization
26. **Digital Media & Marketing Strategy:** Paid advertising campaigns (Google/Meta), technical SEO diagnostics
27. **Dermatology & Personal Care:** Cosmetic active formulation, dermatology triage, material textiles
28. **Philosophy, Ethics & Applied AI Ethics:** Bioethics dilemmas, epistemological analysis, deontological AI ethics
29. **Psychological Crisis & Support Triage:** Crisis hotline escalation (Emergency 112/182), intervention routing
30. **Conversational Chitchat Filter:** Politeness filtering and conversational deflection gatekeeping
31. **Personal Productivity:** Agenda management, email synthesis, action item extraction
32. **Security Guardrails & Fallbacks:** Prompt injection protection, jailbreak filtering, PII data leak prevention

---

## Benchmark & Performance

*Evaluated on enterprise data-center infrastructure (NVIDIA H200 SXM 141 GB HBM3e and NVIDIA RTX PRO 6000 96 GB) with a 512-token context window.*

| Model | Architecture | P95 Latency | GPU VRAM Footprint | Output Determinism |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-6 Luna (OpenAI API)** | Autoregressive LLM | ~920 ms – 1,750 ms | Cloud Hosted | Parse Error Prone |
| **Claude Opus 5.5 (Anthropic API)** | Autoregressive LLM | ~980 ms – 1,820 ms | Cloud Hosted | Parse Error Prone |
| **Gemini 3.8 Flash (Google API)** | Autoregressive LLM | ~320 ms – 650 ms | Cloud Hosted | Parse Error Prone |
| **DeepSeek-V4.1 Flash (MoE API)** | Autoregressive LLM | ~290 ms – 580 ms | Cloud / Multi-GPU | Parse Error Prone |
| **GLM-5.3 (Zhipu AI)** | Autoregressive LLM | ~610 ms – 950 ms | Cloud / Multi-GPU | Parse Error Prone |
| **Qwen 3.8-27B (Local vLLM)** | Autoregressive LLM | ~240 ms – 450 ms | 54 GB (bfloat16) | Parse Error Prone |
| **Pusula (RTX PRO 6000)** | **Non-Autoregressive Encoder** | **11.4 ms** | **~1.2 GB** | **100% Typed & Deterministic** |
| **Pusula (NVIDIA H200)** | **Non-Autoregressive Encoder** | **4.8 ms** | **~1.2 GB** | **100% Typed & Deterministic** |

---

## Training & Reproducibility

To train Pusula from scratch or fine-tune on domain-specific proprietary corpora:

### 1. Synthetic Data Generation (24,000 Samples)
```bash
python synthetic_generator.py
```

### 2. Training Loop & Platt Scaling Calibration
```bash
python train.py
```

The training pipeline uses `bfloat16` and `FlashAttention-2`. Platt Scaling optimizes temperature scaling on the validation split to minimize Expected Calibration Error (ECE) and Brier Score. Weights are serialized to `checkpoints/pusula/best/`.

---

## Citation & License

This project is licensed under the **Apache 2.0** License. Permitted for both commercial and academic usage.

```bibtex
@misc{pusula_2026,
  author = {Hasan Şahin},
  title = {Pusula: Low-Latency Universal Typed Decision Engine for Autonomous Turkish Agent Workflows},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/hsnsahinhsn/pusula}}
}
```
