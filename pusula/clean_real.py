"""Clean and anonymize raw Turkish customer messages into train/eval JSONL on this machine.

Reads CSV or JSONL, repairs common UTF-8 mojibake, drops noise and risky rows, replaces provider placeholder
tags and rule-based personal data, optionally masks person names with a local NER model, then writes disjoint
train and eval splits. Everything runs on this machine; nothing is sent anywhere.

    python -m pusula.clean_real --inp raw.csv --text-field metin --out-dir OUT \\
        --eval-n 500 --strata kanal --id-prefix kurum --seed 0 \\
        [--no-ner] [--ner-model akdeniz27/bert-base-turkish-cased-ner] [--ner-min-score 0.5] [--repair-glued] \\
        [--exclude old_raw.csv:metin --exclude old_clean.jsonl]

With --repair-glued, provider tags glued into words are repaired (see ``repair_glued``) instead of dropping the row.
With --exclude (repeatable, ``PATH[:FIELD]``, field defaults to ``state``), rows already present in earlier raw or clean
files are dropped as ``onceki_veri``, so a new delivery never repeats a message that is already in a train or eval split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import AbstractSet, Callable, Iterator, Sequence

from .anonymize import FIRST, LAST, Anonymizer, fake_tckn, luhn_ok, needs_review

TAG_RE = re.compile(r"<[A-ZÇĞİÖŞÜ_]+>")
# A tag glued to letters ("Ç<AD>ŞAN" for "ÇALIŞAN"): the provider replaced a name inside a word (see repair_glued).
GLUED_TAG = re.compile(r"[^\W\d_]<[A-ZÇĞİÖŞÜ_]+>|<[A-ZÇĞİÖŞÜ_]+>[^\W\d_]")
# "<AD>" followed by more capitalized words or tags is one person ("<AD> Akkaya", "<AD> <AD>"); forms of address stay.
NAME_RUN = re.compile(
    r"<AD>(?:\s+(?:<AD>|(?!(?:Bey|Hanım|Hanımefendi|Hoca|Hocam|Usta|Abla|Abi|Ağabey|Efendi)\b)"
    r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]+))+"
)
# Lead byte of a Turkish letter in UTF-8 (C3/C4/C5) read as latin-1 or cp1252, followed by a continuation byte
# (0x80-0xBF as latin-1, or the cp1252 characters for 0x80-0x9F).
MOJIBAKE_MARKERS = re.compile(
    "[ÃÄÅ][\x80-\xbf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc\u2013\u2014"
    "\u2018-\u201e\u2020-\u2022\u2026\u2030\u2039\u203a\u20ac\u2122]"
)
LETTER_RE = re.compile(r"[^\W\d_]")
WORD_RE = re.compile(r"[^\W\d_]{2,}")

NameFinder = Callable[[list[str]], list[list[tuple[int, int, str, float]]]]

FAKE_ADDRESSES = [
    "Çiçek Sok. No: 12, Kadıköy/İstanbul",
    "Gül Mah. 45/3, Ankara",
    "Lale Cad. No: 7, İzmir",
]
FAKE_COMPANIES = ["Yıldız Teknoloji", "Mavi Kargo", "Ege Sigorta", "Deniz Enerji"]
MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

TAG_HANDLERS: dict[str, str] = {
    "AD": "isim", "ISIM": "isim", "İSİM": "isim",
    "TEL": "telefon", "TELEFON": "telefon",
    "MAIL": "eposta", "EPOSTA": "eposta", "E_POSTA": "eposta",
    "TCKN": "tckn", "KART_NO": "kart", "IBAN": "iban", "ADRES": "adres",
    "TARIH": "tarih", "TARİH": "tarih",
    "SIPARIS_NO": "siparis", "SİPARİŞ_NO": "siparis", "NO": "siparis",
    "KOD": "kod", "KURUM": "kurum", "FIRMA": "kurum", "ŞİRKET": "kurum",
    "PLAKA": "plaka",
}

STREET_ADDR = re.compile(
    r"(?i)\b(?:"
    r"mah\.?|mahallesi|sok\.?|sokak|sokağı|cad\.?|cadde|caddesi|bulvarı|apt\.?|apartmanı"
    r")(?:\s+\S+){0,4}\s+\d"
)
ADDR_NO = re.compile(r"(?i)\bno\s*:\s*\d|\bdaire\s*:?\s*\d|\bkat\s*:\s*\d")
# A title word (whole word, any case) followed by a capitalized word: a real name the NER step missed. Names the NER
# step found are already "<AD>" here, so they do not match.
TITLE_CUE = re.compile(
    r"(?<![^\W\d_])(?i:dr|doç|prof|av|uzm|doktor|hoca|hemşire|müdür|müdire|şef)\.?\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+"
)


def fix_mojibake(text: str) -> str | None:
    """Repair latin-1/cp1252 mis-decoded UTF-8; unchanged if no markers; None if still broken."""
    current = text
    for _ in range(2):
        if not MOJIBAKE_MARKERS.search(current):
            break
        fixed = _decode_mojibake_once(current)
        if fixed is None:
            return None
        current = fixed
    return None if MOJIBAKE_MARKERS.search(current) else current


def _decode_mojibake_once(s: str) -> str | None:
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        buf = bytearray()
        for ch in s:
            try:
                buf.extend(ch.encode("cp1252"))
            except UnicodeEncodeError:
                try:
                    buf.extend(ch.encode("latin-1"))
                except UnicodeEncodeError:
                    return None
        try:
            return bytes(buf).decode("utf-8")
        except UnicodeDecodeError:
            return None


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        if ch in "\n\r":
            out.append(" ")
        elif unicodedata.category(ch).startswith("C"):
            continue
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").casefold()


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


# Common words the provider took for names inside longer words ("Ç<AD>ŞIYORSUN" = ÇALIŞIYORSUN, "<AD>ımız" = canımız).
GLUE_FILLS = ("can", "alı", "ali", "ece", "nur", "naz", "gül", "şen", "han", "ay", "su", "efe", "cem", "ilk", "er")
TAG_GLUE_LEFT_OK = {"KURUM", "ADRES", "AD"}  # a stray word prefix before these is dropped, the placeholder stays


class Vocab:
    """Lower-cased Turkish words seen in clean text; answers 'is this a known word or a prefix of one'."""

    def __init__(self, texts):
        import bisect

        self._bisect = bisect
        self.words = sorted({tr_lower(w) for t in texts for w in WORD_RE.findall(t)})

    def known_prefix(self, s: str) -> int:
        """Length of the longest prefix of ``s`` that starts some known word."""
        lo = 0
        for n in range(len(s), 0, -1):
            p = s[:n]
            i = self._bisect.bisect_left(self.words, p, lo)
            if i < len(self.words) and self.words[i].startswith(p):
                return n
        return 0


def repair_glued(text: str, vocab: Vocab) -> str | None:
    """Repair provider tags glued into words; None if one cannot be repaired (the row is then dropped).

    A tag with letters only before it ("Bu ş<KURUM> raporu") keeps the placeholder and drops the stray prefix.
    An <AD> glued to letters after it is a common word the provider took for a name: the fill from GLUE_FILLS whose
    word is known (or the longest known prefix covering the fill and two more letters) replaces the tag.
    """
    out: list[str] = []
    pos = 0
    for m in TAG_RE.finditer(text):
        tag = m.group(0)[1:-1]
        s, e = m.span()
        ls = s
        while ls > pos and LETTER_RE.match(text[ls - 1]):
            ls -= 1
        re_ = e
        while re_ < len(text) and LETTER_RE.match(text[re_]):
            re_ += 1
        left, right = text[ls:s], text[e:re_]
        if not left and not right:
            continue
        if right and tag == "AD":
            upper = (left + right).isupper()
            best, best_n = None, len(left) + 1
            for fill in GLUE_FILLS:
                word = tr_lower(left + fill + right)
                n = vocab.known_prefix(word)
                need = min(len(word), len(left) + len(fill) + 2)
                if n >= need and n > best_n:
                    best, best_n = fill, n
            if best is None:
                return None
            out.append(text[pos:s] + (tr_upper(best) if upper else best))
            pos = e
        elif left and not right and tag in TAG_GLUE_LEFT_OK:
            out.append(text[pos:ls].rstrip() + (" " if ls > 0 else "") + m.group(0))
            pos = e
        else:
            return None
    out.append(text[pos:])
    fixed = "".join(out)
    return None if GLUED_TAG.search(fixed) else fixed


def dedupe_key(text: str) -> str:
    t = tr_lower(text)
    parts: list[str] = []
    for ch in t:
        if ch.isalnum():
            parts.append(ch)
        elif ch.isspace():
            parts.append(" ")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def is_meaningless(text: str) -> bool:
    stripped = TAG_RE.sub("", text)
    if not LETTER_RE.search(stripped):
        return True
    words = WORD_RE.findall(stripped)
    return len(words) < 2


def has_leftover_risk(text: str) -> bool:
    return bool(STREET_ADDR.search(text) or ADDR_NO.search(text) or TITLE_CUE.search(text))


def _fake_card(rng: random.Random) -> str:
    while True:
        digits = "4" + "".join(str(rng.randint(0, 9)) for _ in range(15))
        if luhn_ok(digits):
            return digits


def _tag_replacement(tag: str, rng: random.Random) -> str:
    kind = TAG_HANDLERS[tag]
    if kind == "isim":
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if kind == "telefon":
        return f"05{rng.randint(30, 59)} {rng.randint(100, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}"
    if kind == "eposta":
        return f"{rng.choice(FIRST).lower()}{rng.randint(10, 99)}@ornek.com.tr"
    if kind == "tckn":
        return fake_tckn(rng)
    if kind == "kart":
        return _fake_card(rng)
    if kind == "iban":
        return "TR" + "".join(str(rng.randint(0, 9)) for _ in range(24))
    if kind == "adres":
        return rng.choice(FAKE_ADDRESSES)
    if kind == "tarih":
        return f"{rng.randint(1, 28)} {rng.choice(MONTHS_TR)}"
    if kind == "siparis":
        return "".join(str(rng.randint(0, 9)) for _ in range(8))
    if kind == "kod":
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(6))
    if kind == "kurum":
        return rng.choice(FAKE_COMPANIES)
    if kind == "plaka":
        return f"{rng.randint(1, 81):02d} {''.join(rng.choice('ABCDEFGHJKLMNPRSTUVYZ') for _ in range(2))} {rng.randint(100, 999)}"
    raise KeyError(tag)


def replace_tags(text: str, rng: random.Random, counts: Counter[str]) -> tuple[str | None, str | None]:
    """Replace provider tags; on unknown tag return (None, reason)."""
    for m in TAG_RE.finditer(text):
        tag = m.group(0)[1:-1]
        if tag not in TAG_HANDLERS:
            return None, "bilinmeyen_etiket"

    def repl(m: re.Match[str]) -> str:
        tag = m.group(0)[1:-1]
        counts[f"etiket_{tag}"] += 1
        return _tag_replacement(tag, rng)

    return TAG_RE.sub(repl, text), None


def _person_label(label: str) -> bool:
    u = label.upper().replace("-", "_")
    return u in {"PER", "PERSON", "B_PER", "I_PER"} or u.endswith("PER")


def _extend_word_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend an entity span to whole words; an apostrophe suffix ("Ayşe'ye") stays outside the span."""
    while start > 0 and LETTER_RE.match(text[start - 1]):
        start -= 1
    while end < len(text) and LETTER_RE.match(text[end]):
        end += 1
    return start, end


def _in_tag(text: str, start: int, end: int) -> bool:
    for m in TAG_RE.finditer(text):
        if start >= m.start() and end <= m.end():
            return True
    return False


def apply_ner(text: str, finder: NameFinder, min_score: float, counts: Counter[str]) -> str:
    """Replace person names found by the NER model with "<AD>"; the tag step later turns it into a fake name.

    No name is skipped: at this point the text holds no fake names yet, so a real "Mehmet" must be replaced too."""
    spans = sorted(
        _extend_word_span(text, s, e) for s, e, lab, sc in finder([text])[0] if _person_label(lab) and sc >= min_score
    )
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in reversed(merged):
        if s < e and not _in_tag(text, s, e):
            text = text[:s] + "<AD>" + text[e:]
            counts["isim_ner"] += 1
    return text


def _chunk_ranges(text: str, max_words: int = 120) -> list[tuple[int, int]]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return [(0, len(text))]
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(words):
        j = min(i + max_words, len(words))
        start = words[i].start()
        end = words[j - 1].end()
        ranges.append((start, end))
        i = j
    return ranges


class HFNameFinder:
    """Local Hugging Face token-classification NER (lazy import so tests stay offline)."""

    def __init__(self, model: str, device: str = "cpu", batch_size: int = 16):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline  # noqa: PLC0415

            self._pipe = pipeline(
                "token-classification",
                model=self.model,
                device=self.device,
                aggregation_strategy="first",
            )
        return self._pipe

    def __call__(self, texts: list[str]) -> list[list[tuple[int, int, str, float]]]:
        pipe = self._load()
        out: list[list[tuple[int, int, str, float]]] = []
        for text in texts:
            ranges = _chunk_ranges(text)
            ents = pipe([text[a:b] for a, b in ranges], batch_size=self.batch_size)
            out.append([
                (a + int(e["start"]), a + int(e["end"]), str(e["entity_group"]), float(e["score"]))
                for (a, _b), chunk_ents in zip(ranges, ents)
                for e in chunk_ents
            ])
        return out


def make_id(prefix: str, text: str, seen: set[str]) -> str:
    base = f"{prefix}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"
    if base not in seen:
        seen.add(base)
        return base
    n = 2
    while f"{base}-{n}" in seen:
        n += 1
    rid = f"{base}-{n}"
    seen.add(rid)
    return rid


def stratified_eval_indices(
    rows: list[dict],
    eval_n: int,
    strata_field: str,
    seed: int,
) -> set[int]:
    n = len(rows)
    cap = min(eval_n, n // 5) if n else 0
    if cap == 0:
        return set()
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = str(row.get("meta", {}).get(strata_field, "") or "")
        groups[key].append(i)
    alloc: dict[str, int] = {k: 0 for k in groups}
    exact = {k: len(v) / n * cap for k, v in groups.items()}
    base_sum = 0
    remainders: list[tuple[float, str]] = []
    for k in groups:
        b = int(exact[k])
        alloc[k] = b
        base_sum += b
        remainders.append((exact[k] - b, k))
    need = cap - base_sum
    remainders.sort(key=lambda x: (-x[0], x[1]))
    for _, k in remainders[:need]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen: set[int] = set()
    for k, idxs in sorted(groups.items()):
        pool = list(idxs)
        rng.shuffle(pool)
        for i in pool[: alloc[k]]:
            chosen.add(i)
    return chosen


def iter_rows(path: Path, text_field: str) -> Iterator[tuple[int, str, dict]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            for row_num, row in enumerate(csv.DictReader(f), start=1):
                text = row.pop(text_field, "") or ""
                meta = dict(row)
                meta["source_row"] = row_num
                yield row_num, text, meta
    else:
        for row_num, line in enumerate(path.open(encoding="utf-8"), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.pop(text_field, "") or "")
            meta = {k: v for k, v in row.items()}
            meta["source_row"] = row_num
            yield row_num, text, meta


def load_exclude_keys(specs: Sequence[str]) -> set[str]:
    """Dedupe keys of every message in earlier files (``PATH[:FIELD]``, FIELD defaults to ``state``). Raw files go
    through the same mojibake repair and normalization as new rows, so a re-delivered raw message matches."""
    keys: set[str] = set()
    for spec in specs:
        path, sep, field = spec.rpartition(":")
        if not sep or "/" in field:
            path, field = spec, "state"
        for _, text, _ in iter_rows(Path(path), field):
            if text:
                keys.add(dedupe_key(_normalize_text(fix_mojibake(text) or text)))
    keys.discard("")
    return keys


def process_rows(
    items: list[tuple[int, str, dict]],
    *,
    seed: int,
    id_prefix: str,
    eval_n: int,
    strata: str,
    use_ner: bool,
    name_finder: NameFinder | None,
    ner_min_score: float,
    repair_vocab: Vocab | None = None,
    exclude_keys: AbstractSet[str] = frozenset(),
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Run the cleaning pipeline; returns train, eval, dropped, review rows, and report dict."""
    anon = Anonymizer(seed)
    seen_keys: set[str] = set()
    dropped: list[dict] = []
    clean: list[dict] = []
    review_rows: list[dict] = []
    stats: Counter[str] = Counter()
    mojibake_fixed = 0
    tag_counts: Counter[str] = Counter()
    ner_counts: Counter[str] = Counter()
    anon_counts: Counter[str] = Counter()

    finder = name_finder

    for _row_num, raw_text, meta in items:
        text = fix_mojibake(raw_text)
        if text is None:
            dropped.append({"source_row": meta["source_row"], "reason": "kodlama", "text": raw_text})
            stats["kodlama"] += 1
            continue
        if text != raw_text:
            mojibake_fixed += 1
        text = _normalize_text(text)

        if is_meaningless(text):
            dropped.append({"source_row": meta["source_row"], "reason": "anlamsiz", "text": raw_text})
            stats["anlamsiz"] += 1
            continue

        dkey = dedupe_key(text)
        if dkey in exclude_keys:
            dropped.append({"source_row": meta["source_row"], "reason": "onceki_veri", "text": raw_text})
            stats["onceki_veri"] += 1
            continue
        if dkey in seen_keys:
            dropped.append({"source_row": meta["source_row"], "reason": "tekrar", "text": raw_text})
            stats["tekrar"] += 1
            continue
        seen_keys.add(dkey)

        row_rng = random.Random(seed ^ meta["source_row"])

        if GLUED_TAG.search(text):
            fixed = repair_glued(text, repair_vocab) if repair_vocab is not None else None
            if fixed is None:
                dropped.append({"source_row": meta["source_row"], "reason": "bozuk_etiket", "text": raw_text})
                stats["bozuk_etiket"] += 1
                continue
            text = fixed
            stats["etiket_onarildi"] += 1

        if use_ner and finder is not None:
            text = apply_ner(text, finder, ner_min_score, ner_counts)
        text = NAME_RUN.sub("<AD>", text)

        if has_leftover_risk(text):
            dropped.append({"source_row": meta["source_row"], "reason": "kisisel_veri_supheli", "text": raw_text})
            stats["kisisel_veri_supheli"] += 1
            continue

        text, tag_err = replace_tags(text, row_rng, tag_counts)
        if tag_err:
            dropped.append({"source_row": meta["source_row"], "reason": tag_err, "text": raw_text})
            stats[tag_err] += 1
            continue

        text = anon(text)
        for k, v in anon.counts.items():
            anon_counts[k] += v
        anon.counts.clear()

        if exclude_keys and dedupe_key(text) in exclude_keys:  # earlier clean files hold the anonymized text
            dropped.append({"source_row": meta["source_row"], "reason": "onceki_veri", "text": raw_text})
            stats["onceki_veri"] += 1
            continue

        words = needs_review(text)
        clean.append({"text": text, "meta": meta, "_dedupe": dkey, "_review": words})

    eval_idx = stratified_eval_indices(clean, eval_n, strata, seed)
    seen_ids: set[str] = set()
    train: list[dict] = []
    eval_rows: list[dict] = []
    train_strata: Counter[str] = Counter()
    eval_strata: Counter[str] = Counter()

    for i, rec in enumerate(clean):
        rid = make_id(id_prefix, rec["text"], seen_ids)
        row = {"id": rid, "state": rec["text"], "meta": rec["meta"]}
        st = str(rec["meta"].get(strata, "") or "")
        if rec["_review"]:
            review_rows.append({"id": rid, "words": rec["_review"], "text": rec["text"]})
        if i in eval_idx:
            eval_rows.append(row)
            eval_strata[st] += 1
        else:
            train.append(row)
            train_strata[st] += 1

    train_keys = {dedupe_key(r["state"]) for r in train}
    eval_keys = {dedupe_key(r["state"]) for r in eval_rows}
    assert not (set(r["id"] for r in train) & set(r["id"] for r in eval_rows))
    assert not (train_keys & eval_keys)

    replacements = dict(tag_counts)
    replacements.update(ner_counts)
    replacements.update(anon_counts)

    report = {
        "input": len(items),
        "clean": len(clean),
        "train": len(train),
        "eval": len(eval_rows),
        "dropped": dict(stats),
        "mojibake_fixed": mojibake_fixed,
        "train_strata": dict(train_strata),
        "eval_strata": dict(eval_strata),
        "replacements": replacements,
        "ner_enabled": use_ner,
        "ner_model": None,
    }
    return train, eval_rows, dropped, review_rows, report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def run(
    inp: Path,
    text_field: str,
    out_dir: Path,
    eval_n: int,
    strata: str,
    id_prefix: str,
    seed: int,
    use_ner: bool,
    ner_model: str,
    ner_min_score: float,
    name_finder: NameFinder | None = None,
    repair: bool = False,
    exclude: Sequence[str] = (),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o700)

    finder = name_finder
    if use_ner and finder is None:
        finder = HFNameFinder(ner_model)

    items = list(iter_rows(inp, text_field))
    vocab = Vocab(_normalize_text(fix_mojibake(t) or t) for _, t, _ in items) if repair else None
    train, eval_rows, dropped, review_rows, report = process_rows(
        items,
        seed=seed,
        id_prefix=id_prefix,
        eval_n=eval_n,
        strata=strata,
        use_ner=use_ner,
        name_finder=finder if use_ner else None,
        ner_min_score=ner_min_score,
        repair_vocab=vocab,
        exclude_keys=load_exclude_keys(exclude),
    )
    if use_ner and name_finder is None:
        report["ner_model"] = ner_model

    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    _write_jsonl(out_dir / "dropped.jsonl", dropped)

    with (out_dir / "review.tsv").open("w", encoding="utf-8") as f:
        f.write("id\twords\ttext\n")
        for rev in review_rows:
            rid = rev.get("id") or ""
            words = ", ".join(rev["words"])
            snippet = rev["text"][:300].replace("\t", " ")
            f.write(f"{rid}\t{words}\t{snippet}\n")
    (out_dir / "review.tsv").chmod(0o600)

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    (out_dir / "report.json").chmod(0o600)

    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inp", required=True, type=Path)
    p.add_argument("--text-field", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--eval-n", type=int, default=500)
    p.add_argument("--strata", default="kanal")
    p.add_argument("--id-prefix", default="kurum")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ner", action="store_true")
    p.add_argument("--ner-model", default="akdeniz27/bert-base-turkish-cased-ner")
    p.add_argument("--ner-min-score", type=float, default=0.5)
    p.add_argument("--repair-glued", action="store_true", help="repair tags glued into words instead of dropping the row")
    p.add_argument("--exclude", action="append", default=[], metavar="PATH[:FIELD]",
                   help="drop rows already in this earlier raw or clean file (repeatable; FIELD defaults to state)")
    a = p.parse_args(argv)
    report = run(
        a.inp,
        a.text_field,
        a.out_dir,
        a.eval_n,
        a.strata,
        a.id_prefix,
        a.seed,
        use_ner=not a.no_ner,
        ner_model=a.ner_model,
        ner_min_score=a.ner_min_score,
        repair=a.repair_glued,
        exclude=a.exclude,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
