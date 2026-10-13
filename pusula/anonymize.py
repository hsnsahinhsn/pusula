"""Local anonymization of real Turkish messages before they are labeled or evaluated.

Personal data is replaced with synthetic values of the same kind (a valid-looking but fake phone number, a fake
T.C. kimlik no with a correct checksum, a fake name...), not with tags: a message that contained personal data still
reads like one, so the `kisisel_veri` question keeps its meaning. No mapping back to the original is stored.
Everything runs on this machine; nothing is sent anywhere.

    python -m pusula.anonymize --inp raw.jsonl --text-field text --out anon.jsonl --review review.tsv

Detected: e-mail, phone, T.C. kimlik no, IBAN, card number (Luhn), licence plate, URL, @handle, long digit runs
(order/customer/student numbers) and names after a cue word ("adım", "ben", "Sayın", "... Bey/Hanım"). Names elsewhere
and street addresses cannot be found reliably by rules: every line where capitalized words remain is written to the
review file for a person to check before the data leaves this machine.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

FIRST = ["Ayşe", "Mehmet", "Zeynep", "Mustafa", "Elif", "Ahmet", "Fatma", "Emre", "Selin", "Burak", "Deniz", "Can"]
LAST = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Aydın", "Öztürk", "Arslan", "Doğan", "Kılıç", "Koç"]

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
IBAN = re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.I)
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?90[\s-]?)?\(?0?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
                   r"|(?<!\d)(?:\+?90[\s-]?)?\(?0?[234]\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
TCKN = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
PLATE = re.compile(r"\b(0[1-9]|[1-7]\d|8[01])\s?[A-ZÇĞİÖŞÜ]{1,3}\s?\d{2,4}\b")
HANDLE = re.compile(r"(?<![\w@])@[A-Za-z0-9_.]{2,30}\b")
DIGITS = re.compile(r"(?<!\d)(?<!\d[.,])\d{6,}(?!\d|[.,]\d)")  # not inside 1.234.567 or 12,5; trailing punctuation is fine
NAME_CUE = re.compile(r"(?i:\b(?:adım|ismim|benim adım|ben)\s+)([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?)"
                      r"|(?i:\bsayın\s+)([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?)"
                      r"|\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+(?:Bey|Hanım|bey|hanım)\b")


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def tckn_ok(s: str) -> bool:
    d = [int(c) for c in s]
    if len(d) != 11 or d[0] == 0:
        return False
    d10 = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
    return d[9] == d10 and d[10] == sum(d[:10]) % 10


def fake_tckn(rng: random.Random) -> str:
    while True:
        d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
        d10 = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
        d.append(d10)
        d.append(sum(d) % 10)
        s = "".join(map(str, d))
        if tckn_ok(s):
            return s


class Anonymizer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.counts: dict[str, int] = {}

    def _count(self, kind: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _name(self, full: bool) -> str:
        first = self.rng.choice(FIRST)
        return f"{first} {self.rng.choice(LAST)}" if full else first

    def __call__(self, text: str) -> str:
        r = self.rng

        def sub(pattern, kind, make, check=None):
            nonlocal text

            def repl(m):
                raw = m.group(0)
                if check and not check(re.sub(r"\D", "", raw)):
                    return raw
                self._count(kind)
                return make(m)
            text = pattern.sub(repl, text)

        sub(EMAIL, "eposta", lambda m: f"{self._name(False).lower()}{r.randint(10, 99)}@ornek.com.tr")
        sub(URL, "url", lambda m: "https://ornek.com.tr/sayfa")
        sub(IBAN, "iban", lambda m: "TR" + "".join(str(r.randint(0, 9)) for _ in range(24)))
        sub(CARD, "kart", lambda m: "4" + "".join(str(r.randint(0, 9)) for _ in range(15)), luhn_ok)
        sub(TCKN, "tckn", lambda m: fake_tckn(r), tckn_ok)
        sub(PHONE, "telefon", lambda m: f"05{r.randint(30, 59)} {r.randint(100, 999)} {r.randint(10, 99)} {r.randint(10, 99)}")
        sub(PLATE, "plaka", lambda m: f"{r.randint(1, 81):02d} {''.join(r.choice('ABCDEFGHJKLMNPRSTUVYZ') for _ in range(2))} {r.randint(100, 999)}")
        sub(HANDLE, "kullanici_adi", lambda m: f"@kullanici{r.randint(100, 999)}")
        sub(DIGITS, "numara", lambda m: "".join(str(r.randint(0, 9)) for _ in m.group(0)))

        def name_repl(m):
            self._count("isim")
            name = next(g for g in m.groups() if g)
            return m.group(0).replace(name, self._name(" " in name))
        text = NAME_CUE.sub(name_repl, text)
        return text


def needs_review(text: str) -> list[str]:
    """Capitalized words that are not at a sentence start: possible names, places or companies a person should check."""
    words = re.findall(r"(?<![.!?\n]\s)(?<!^)\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b", text)
    known = set(FIRST) | set(LAST)
    return sorted({w for w in words if w not in known})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inp", required=True, help="JSONL with one message per line")
    p.add_argument("--text-field", default="text")
    p.add_argument("--out", required=True)
    p.add_argument("--review", required=True, help="TSV of lines a person must check (possible names, places)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    anon = Anonymizer(a.seed)
    n = flagged = 0
    with open(a.out, "w", encoding="utf-8") as out, open(a.review, "w", encoding="utf-8") as rev:
        rev.write("satir\tkontrol_edilecek_kelimeler\tmetin\n")
        for i, line in enumerate(open(a.inp, encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row[a.text_field] = anon(row[a.text_field])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            words = needs_review(row[a.text_field])
            if words:
                flagged += 1
                rev.write(f"{i}\t{', '.join(words)}\t{row[a.text_field][:300].replace(chr(9), ' ')}\n")
    print(json.dumps({"mesaj": n, "insan_kontrolu": flagged, "degistirilen": anon.counts}, ensure_ascii=False))
    Path(a.review).touch()


if __name__ == "__main__":
    main()
