# Turkish Decision Benchmark (v1)

`tr_decision_bench_v1.jsonl` — 391 realistic Turkish messages with 3,441 labeled decision questions,
for measuring how well a decision model (Decision 3.0, Pusula, or any other) handles Turkish.

- **Coverage:** all 20 domains of the Pusula taxonomy (18–20 messages each), 299 intents available.
- **Variety:** short chat lines, formal e-mails and petitions, angry complaints, voice dictation without
  punctuation, call-center transcripts, internal tickets and log snippets; typos, missing Turkish
  characters, abbreviations and Turkish–English code-switching; embedded prompt-injection and jailbreak
  attempts; fake personal data.
- **Questions per message:** `alan` / `alt_alan` / `niyet` (domain → subdomain → intent routing), the six
  universal questions (`aciliyet`, `eylem_gerekli`, `insana_devret`, `guvenlik`, `duygu`, `kisisel_veri`)
  and, for 115 messages, an ad-hoc business question with its own options.
- **Labels:** every message was written and labeled by one annotator and labeled again blind by a second
  one. Only messages whose routing labels agree are included, and each other question is kept only where
  both annotators gave the same answer.

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
