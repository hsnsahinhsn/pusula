"""Stage 1: generate Turkish decision examples with a local LLM (Qwen3.8-27B on vLLM).

Label-first generation: each request fixes a target path in the taxonomy (alan > alt_alan > niyet)
plus a writing style, persona, length and some universal targets; the LLM writes a realistic
message for it and labels the remaining universal questions. A share of requests also asks for an
ad-hoc business question with its own options, so the student learns to read options it has never
seen. Output is resumable: re-running continues until --n examples exist.

    python -m pusula.generate --n 30000 --out runs/full/generated.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .jsonl import JsonlWriter, read_jsonl
from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_json, console, make_client, run_pool, wait_for_server
from .schema import Example, Question, stable_id
from .taxonomy import IntentPath, Taxonomy, load_taxonomy

ADHOC_QUESTION = "ozel_soru"

STYLES = [
    "kısa ve aceleyle yazılmış tek cümlelik mesaj",
    "günlük konuşma dilinde birkaç cümlelik mesaj",
    "yazım hataları, kısaltmalar ve Türkçe karakter eksikliği içeren mesaj (ör. 'kartim calindi acil yardim')",
    "resmi ve kurumsal bir e-posta (selamlama, gövde ve imza ile)",
    "noktalaması özensiz, aklına geldiği gibi yazılmış mesaj",
    "arka plan hikayesiyle başlayıp asıl talebi sonda söyleyen uzun mesaj",
    "çağrı merkezi görüşmesinin dökümü ('Müşteri:' ve 'Temsilci:' satırlarıyla)",
    "sistem logu, hata kodu veya JSON parçası içeren teknik bildirim",
    "sesli asistana söylenmiş gibi, noktalamasız konuşma dili",
    "bir çalışanın kurum içi talep sistemine yazdığı kısa iş kaydı",
    "WhatsApp mesajı tarzında, emoji içerebilen kısa mesaj",
    "birkaç konuya değinen ama asıl talebi belirgin olan mesaj",
    "önceki yazışmaya yanıt olarak gönderilmiş, alıntı içeren e-posta",
    "yabancı kökenli kelimeler ve İngilizce terimlerle karışık Türkçe mesaj",
]

SHORT_STYLES = {STYLES[0], STYLES[2], STYLES[9], STYLES[10]}

PERSONAS = [
    "üniversite öğrencisi", "emekli ve teknolojiyle arası iyi olmayan bir vatandaş", "KOBİ sahibi",
    "büyük bir şirkette çalışan beyaz yaka", "serbest muhasebeci", "yazılım geliştirici",
    "saha teknisyeni", "hasta yakını", "çiftçi", "e-ticaret satıcısı", "lise öğrencisinin velisi",
    "kurumsal satın alma uzmanı", "lojistik operasyon sorumlusu", "ev hanımı", "esnaf",
    "insan kaynakları uzmanı", "taşrada yaşayan genç", "yurt dışında yaşayan Türk vatandaşı",
]

LENGTHS = [
    ("kısa: 1-2 cümle", 0.30),
    ("orta: 3-5 cümle", 0.35),
    ("uzun: 6-12 cümle", 0.25),
    ("çok uzun: 250-500 kelime, gerçekçi ayrıntılarla", 0.10),
]

DUYGU_WEIGHTS = {
    "notr": 0.28, "memnun": 0.08, "endiseli": 0.18, "uzgun": 0.06, "hayal_kirikligi": 0.17,
    "ofkeli": 0.13, "caresiz_kriz": 0.10,
}


@dataclass
class Spec:
    index: int
    path: IntentPath
    style: str
    persona: str
    length: str
    duygu: str
    guvenlik: str
    kisisel_veri: bool
    confusable: Optional[str] = None
    adhoc: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _weighted(rng: random.Random, pairs) -> Any:
    items, weights = zip(*pairs)
    return rng.choices(items, weights=weights, k=1)[0]


def make_specs(tax: Taxonomy, start: int, count: int, seed: int, adhoc_rate: float = 0.25,
               hard_rate: float = 0.2, unsafe_rate: float = 0.12) -> list[Spec]:
    """Balanced over intents: every intent appears once per cycle, in a per-cycle shuffled order."""
    paths = list(tax.intent_paths())
    guv_keys = [k for k in tax.universal["guvenlik"].option_keys() if k != "guvenli"]
    duygu_keys = [k for k in tax.universal["duygu"].option_keys() if k in DUYGU_WEIGHTS]
    specs = []
    for idx in range(start, start + count):
        cycle, pos = divmod(idx, len(paths))
        order = list(range(len(paths)))
        random.Random(seed * 1_000_003 + cycle).shuffle(order)
        path = paths[order[pos]]
        rng = random.Random(seed * 7919 + idx)
        sub = tax.domains[path.domain].subdomains[path.subdomain]
        siblings = [k for k in sub.intents if k != path.intent]
        confusable = rng.choice(siblings) if siblings and rng.random() < hard_rate else None
        guv = rng.choice(guv_keys) if rng.random() < unsafe_rate else "guvenli"
        crisis = "caresiz_kriz" in duygu_keys and path.domain == "kriz_destek" and rng.random() < 0.6  # taxonomy v1 only
        duygu = "caresiz_kriz" if crisis else _weighted(rng, [(k, DUYGU_WEIGHTS[k]) for k in duygu_keys])
        style = rng.choice(STYLES)
        # Short-form styles cannot be 500 words long; keep style and length consistent.
        lengths = LENGTHS[:2] if style in SHORT_STYLES else LENGTHS
        specs.append(Spec(
            index=idx, path=path, style=style, persona=rng.choice(PERSONAS),
            length=_weighted(rng, lengths), duygu=duygu, guvenlik=guv,
            kisisel_veri=rng.random() < 0.25, confusable=confusable,
            adhoc=rng.random() < adhoc_rate,
        ))
    return specs


def _universal_schema(tax: Taxonomy) -> dict[str, Any]:
    props = {}
    for name, q in tax.universal.items():
        if q.type == "noul":
            props[name] = {"type": "boolean"}
        elif q.type == "score":
            props[name] = {"type": "integer", "enum": list(range(len(q.criteria)))}
        else:
            props[name] = {"type": "string", "enum": q.option_keys()}
    return props


def output_schema(tax: Taxonomy, adhoc: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "senaryo": {"type": "string"},  # brief scene-setting before writing; not stored
        "state": {"type": "string", "minLength": 10},
    }
    props.update(_universal_schema(tax))
    if adhoc:
        props[ADHOC_QUESTION] = {
            "type": "object",
            "properties": {
                "soru": {"type": "string"},
                "secenekler": {
                    "type": "array", "minItems": 3, "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {"anahtar": {"type": "string"}, "aciklama": {"type": "string"}},
                        "required": ["anahtar", "aciklama"], "additionalProperties": False,
                    },
                },
                "cevap": {"type": "string"},
            },
            "required": ["soru", "secenekler", "cevap"], "additionalProperties": False,
        }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _describe_universal(tax: Taxonomy) -> str:
    lines = []
    for name, q in tax.universal.items():
        if q.type == "noul":
            fmt = "true/false"
            opts = f"false = {q.option_texts()[0]}; true = {q.option_texts()[1]}"
        elif q.type == "score":
            fmt = f"tam sayı 0-{len(q.criteria) - 1}"
            opts = "; ".join(f"{i} = {lvl}" for i, lvl in enumerate(q.criteria))
        else:
            fmt = "anahtar"
            opts = "; ".join(f"{k} = {v}" for k, v in q.criteria.items())
        lines.append(f"- {name} ({fmt}): {q.instructions} [{opts}]")
    return "\n".join(lines)


def build_system(tax: Taxonomy) -> str:
    """Static instructions, identical for every request so vLLM's prefix cache shares them."""
    return "\n".join([
        "Sen Türkçe karar modelleri için gerçekçi eğitim verisi üreten bir uzmansın. Her istekte verilen "
        "hedefe uygun TEK bir kullanıcı mesajı yazar ve etiketlersin. Sadece istenen JSON nesnesini döndür.",
        "",
        "Yazım kuralları:",
        "1. Önce 'senaryo' alanında 1-2 cümleyle kimin, hangi somut durumda, neden yazdığını kurgula; "
        "sonra 'state' alanına bu kişinin gerçekten göndereceği mesajı yaz.",
        "2. Mesaj tek, somut ve tutarlı bir durumu anlatsın; birbiriyle ilgisiz konuları karıştırma.",
        "3. Mesajın asıl amacı verilen niyet olsun; okuyan bir uzman niyeti mesajdan çıkarabilsin, ama "
        "niyetin adını veya taksonomi terimlerini olduğu gibi kopyalama.",
        "4. Doğal ve özgün yaz; 'Merhaba, size ... hakkında yazıyorum' gibi klişelerden ve şablon "
        "cümlelerden kaçın. Verilen üslup, uzunluk ve duyguya uy; duyguyu üsluba yansıt.",
        "5. Numaralar ve kimlikler bağlama uygun olsun (sipariş no'su telefon numarasına benzemesin). "
        "Kişisel veri gerekiyorsa tamamen uydurma değerler kullan.",
        "6. Güvenlik riski istenirse girişimi normal bir talebin içine doğal biçimde yerleştir; gerçekten "
        "zararlı talimat veya içerik yazma, girişimi göstermek yeterli.",
        "",
        "Mesajı yazdıktan sonra aşağıdaki evrensel soruları, mesajı okuyan tarafsız bir uzman gibi "
        "yanıtla (hedeflerle çelişirse yazdığın mesajın kendisine göre yanıtla):",
        _describe_universal(tax),
        "",
        f"'{ADHOC_QUESTION}' istenirse: bu mesajı işleyen bir şirketin otomasyonunun sorabileceği özgün "
        "bir karar sorusu üret. 'soru' Türkçe bir soru cümlesi; 'secenekler' 3-6 adet, her biri küçük "
        "harfli ASCII snake_case 'anahtar' ve kısa Türkçe 'aciklama' içersin; seçenekler birbirini "
        "dışlasın ve en az biri güçlü bir çeldirici olsun; 'cevap' doğru seçeneğin anahtarı.",
    ])


def build_prompt(tax: Taxonomy, spec: Spec) -> str:
    d = tax.domains[spec.path.domain]
    s = d.subdomains[spec.path.subdomain]
    parts = [
        f"Alan: {d.name} — {d.description}",
        f"Alt alan: {s.name} — {s.description}",
        f"Niyet: {s.intents[spec.path.intent]}",
        f"Yazan kişi: {spec.persona}",
        f"Üslup: {spec.style}",
        f"Uzunluk: {spec.length}",
        f"Baskın duygu: {tax.universal['duygu'].criteria[spec.duygu]}",
    ]
    if spec.confusable:
        parts.append(
            f"Zorluk: mesaj, '{s.intents[spec.confusable]}' niyetiyle karıştırılabilecek kadar yakın "
            "olsun, ancak dikkatli okuyan biri asıl niyeti yine de doğru anlayabilsin."
        )
    if spec.guvenlik != "guvenli":
        parts.append(f"Güvenlik riski: {tax.universal['guvenlik'].criteria[spec.guvenlik]}")
    if spec.kisisel_veri:
        parts.append("Kişisel veri: mesaj uydurma kişisel veri (TC kimlik no, telefon, IBAN veya adres) içersin.")
    parts.append(f"{ADHOC_QUESTION}: {'üret' if spec.adhoc else 'üretme'}")
    return "\n".join(parts)


_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def to_example(tax: Taxonomy, spec: Spec, out: dict[str, Any], model: str) -> Optional[Example]:
    state = (out.get("state") or "").strip()
    if len(state) < 10:
        return None
    questions: dict[str, Question] = tax.questions_for(spec.path)
    labels = spec.path.as_labels()
    for name, q in tax.universal.items():
        v = out[name]
        if q.type == "noul":
            labels[name] = "true" if v else "false"
        else:
            labels[name] = str(v)
    if spec.adhoc and out.get(ADHOC_QUESTION):
        adhoc = out[ADHOC_QUESTION]
        criteria = {}
        for opt in adhoc["secenekler"]:
            key = opt["anahtar"].strip().lower()
            if _KEY_RE.match(key) and key not in criteria and opt["aciklama"].strip():
                criteria[key] = opt["aciklama"].strip()
        answer = adhoc["cevap"].strip().lower()
        if len(criteria) >= 3 and answer in criteria and adhoc["soru"].strip():
            questions[ADHOC_QUESTION] = Question(
                type="choice", instructions=adhoc["soru"].strip(), criteria=criteria)
            labels[ADHOC_QUESTION] = answer
    return Example(
        id=stable_id(state),
        state=state,
        questions=questions,
        meta={
            "path": str(spec.path), "spec_index": spec.index, "style": spec.style,
            "persona": spec.persona, "length": spec.length, "hard": spec.confusable is not None,
            "target_guvenlik": spec.guvenlik, "generator_model": model,
        },
        labels={"generator": labels},
    )


async def run(args) -> None:
    tax = load_taxonomy(args.taxonomy)
    existing = list(read_jsonl(args.out))
    seen = {row["id"] for row in existing}
    max_index = max((row["meta"]["spec_index"] for row in existing), default=-1)
    need = args.n - len(existing)
    if need <= 0:
        console.print(f"[green]{args.out}: zaten {len(existing)} örnek var, üretim atlandı.[/green]")
        return
    await wait_for_server(args.base_url)
    client = make_client(args.base_url)
    # Over-provision specs: some requests fail validation or duplicate an earlier state.
    specs = make_specs(tax, max_index + 1, int(need * 1.15) + 50, args.seed,
                       adhoc_rate=args.adhoc_rate)
    console.print(f"[cyan]Üretim: hedef {args.n}, mevcut {len(existing)}, istek {len(specs)} "
                  f"({len(list(tax.intent_paths()))} niyet)[/cyan]")
    system = build_system(tax)
    writer = JsonlWriter(args.out)
    written = 0
    stop = asyncio.Event()

    async def worker(spec: Spec):
        if stop.is_set():
            return None
        out = await chat_json(
            client, args.model,
            [{"role": "system", "content": system}, {"role": "user", "content": build_prompt(tax, spec)}],
            output_schema(tax, spec.adhoc), temperature=args.temperature, max_tokens=args.max_tokens,
        )
        return to_example(tax, spec, out, args.model) if out else None

    def on_result(spec: Spec, ex: Example):
        nonlocal written
        if ex.id in seen or stop.is_set():
            return
        seen.add(ex.id)
        writer.write(ex)
        written += 1
        if len(existing) + written >= args.n:
            stop.set()

    try:
        await run_pool(specs, worker, args.concurrency, on_result, label="üretim", total=len(specs))
    finally:
        writer.close()
    console.print(f"[green]✓ {written} yeni örnek yazıldı → {args.out}[/green]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True, help="toplam hedef örnek sayısı")
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--taxonomy", default=None)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--adhoc-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=13)
    asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
