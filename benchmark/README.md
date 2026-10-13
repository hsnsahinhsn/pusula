# Turkish Decision Benchmark

`tr_decision_bench_v1.jsonl` — 2,194 realistic Turkish messages with about 20,000 labeled decision questions, for
measuring how well a decision model (Decision 3.0, Pusula, or any other) handles Turkish.

- **Coverage:** all 20 domains of the Pusula taxonomy and its 299 intents; 12 writing styles (formal e-mails,
  everyday chat, no Turkish characters and typos, angry capitals, long and rambling, one short line, Turkish–English
  code-switching, an elderly user's roundabout politeness, young and emoji-heavy, several issues with the real request
  last, error codes and logs, voice dictation without punctuation); about 10% security probes (prompt injection,
  jailbreak, unauthorized access, harmful requests, spam); fake personal data in about 30%.
- **Urgency is about the situation, not the tone:** "acil" appears in 696 messages, and 55% of those are routine or
  can wait a few days.
- **Questions per message:** `alan` / `alt_alan` / `niyet` (domain → subdomain → intent routing), the six universal
  questions (`aciliyet`, `eylem_gerekli`, `insana_devret`, `guvenlik`, `duygu`, `kisisel_veri`) and, for some messages,
  an ad-hoc business question with its own options.
- **Labels:** two annotators answered every question blind and independently; where they disagreed, a third, more
  careful pass decided. All of them follow one written [labeling policy](../pusula/data/label_policy.md), which sets
  the boundaries of the judgment questions (when urgency is high, when to hand off to a person, when an action is
  needed). Annotator agreement before adjudication, i.e. how ambiguous each question is:

  | question | agreement |
  |---|---|
  | alan / alt_alan / niyet | 97.5 / 96.6 / 94.6 |
  | aciliyet | 85.2 |
  | eylem_gerekli | 92.7 |
  | insana_devret | 87.2 |
  | guvenlik | 99.2 |
  | duygu | 90.2 |
  | kisisel_veri | 96.8 |

- **How it was made:** messages were written from randomized recipes (intent × style × length × situation) by Claude
  (Anthropic) and labeled with Claude models, for evaluation only; near-duplicates were removed. No real people,
  companies or personal data. Messages that the annotation model declined to label (a handful of harmful-content
  probes) are not included.

## Results

Accuracy (%) on this benchmark (19,861 questions):

| Model | Accuracy |
|---|---|
| Decision 3.0 (`vllm-sr/d3`) | 88.5 |
| **Pusula** | **92.8** |
| Pusula Nano | 81.8 |
| Pusula Mini | 80.9 |

## Format

One JSON object per line, in the Pusula / d3 request format:

```json
{
  "id": "…",
  "state": "Turkish message",
  "questions": {"alan": {"type": "choice", "instructions": "…", "criteria": {"…": "…"}}, "…": {}},
  "labels": {"gold": {"alan": "finans_bankacilik", "aciliyet": "2", "eylem_gerekli": "true", "…": "…"}}
}
```

Each question is a complete d3 `systemone` question, so any d3-compatible model can be scored by sending
`{"state", "questions"}` and comparing its answers with `labels.gold`
(`choice` → option key, `noul` → `"true"`/`"false"`, `score` → level index as a string).

```bash
python -m pusula.d3train eval --d3 ~/d3 --data benchmark/tr_decision_bench_v1.jsonl --out base_d3.json
```

Please do not train on this benchmark.
