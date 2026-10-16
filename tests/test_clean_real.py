import csv
import json
import random
import stat
from collections import Counter
from pathlib import Path

from pusula.clean_real import (
    TAG_RE,
    apply_ner,
    dedupe_key,
    fix_mojibake,
    has_leftover_risk,
    is_meaningless,
    iter_rows,
    main,
    process_rows,
    replace_tags,
    stratified_eval_indices,
)

TURKISH = "Çok güzel, İstanbul'da ışık; Ğ ğ Ş ş Ö ö Ü ü Ç ç I ı İ i, şikâyet € ‘tırnak’"


def _as_cp1252(s: str) -> str:
    """UTF-8 bytes read as cp1252, with latin-1 for the five bytes cp1252 leaves undefined."""
    return "".join(bytes([b]).decode("cp1252", errors="ignore") or chr(b) for b in s.encode("utf-8"))


def test_mojibake_examples():
    assert fix_mojibake("bÃ¶lÃ¼mÃ¼nde") == "bölümünde"
    assert fix_mojibake(TURKISH.encode("utf-8").decode("latin-1")) == TURKISH
    assert fix_mojibake(_as_cp1252(TURKISH)) == TURKISH
    twice = TURKISH.encode("utf-8").decode("latin-1").encode("utf-8").decode("latin-1")
    assert fix_mojibake(twice) == TURKISH
    for clean in (TURKISH, "Çok güzel", "ÇOK KÖTÜ, İADE İSTİYORUM"):
        assert fix_mojibake(clean) == clean


def test_mojibake_unfixable():
    assert fix_mojibake("Ã¶Ã¼Ã§ broken Ãx") is None


def test_meaningless_and_duplicates():
    assert is_meaningless("!!!")
    assert is_meaningless("<AD> <TEL>")
    assert not is_meaningless("çok kötü kokuyor")
    assert not is_meaningless("Asla telefona bakmıyorlar")
    assert not is_meaningless("şikâyetim büyük")
    assert is_meaningless("rezalet!!!") and is_meaningless("⭐⭐⭐ 5/5")
    assert dedupe_key("Çok kötü!") == dedupe_key("çok kötü")
    assert dedupe_key("İyi değil") == dedupe_key("iyi değil")


def test_tags_replaced_and_unknown_dropped():
    rng = random.Random(0)
    counts: Counter[str] = Counter()
    text, err = replace_tags("Merhaba <AD> aradı <TEL>.", rng, counts)
    assert err is None and text
    assert not TAG_RE.search(text)
    assert counts["etiket_AD"] == 1
    bad, err2 = replace_tags("x <BILINMEYEN> y", rng, counts)
    assert bad is None and err2 == "bilinmeyen_etiket"


def test_ner_replacement_and_chunks():
    def finder(texts: list[str]):
        out = []
        for text in texts:
            spans = []
            for name in ("Funda",):
                pos = text.find(name)
                if pos >= 0:
                    spans.append((pos, pos + len(name), "PER", 0.99))
            out.append(spans)
        return out

    counts: Counter[str] = Counter()
    assert apply_ner("Funda'ya ulaştım", finder, 0.5, counts) == "<AD>'ya ulaştım"
    assert counts["isim_ner"] == 1

    # Names that look like the fake lists are real here and must be replaced too; subword spans grow to words.
    real = "Mehmet Kaya ve Ahmet geldi"
    spans = [[(0, 6, "B-PER", 0.9), (7, 11, "I-PER", 0.9), (15, 18, "PER", 0.8), (19, 24, "PER", 0.3)]]
    assert apply_ner(real, lambda texts: spans, 0.5, counts) == "<AD> <AD> ve <AD> geldi"
    assert apply_ner("<AD> aradı", lambda texts: [[(1, 3, "PER", 0.9)]], 0.5, counts) == "<AD> aradı"


def test_name_finder_chunks_long_text():
    from pusula.clean_real import HFNameFinder

    text = " ".join(["kelime"] * 400) + " Funda geldi"

    def pipe(chunks, batch_size):
        return [[{"start": c.find("Funda"), "end": c.find("Funda") + 5, "entity_group": "PER", "score": 0.99}]
                if "Funda" in c else [] for c in chunks]

    f = HFNameFinder("yok")
    f._pipe = pipe
    (spans,) = f([text])
    assert [text[s:e] for s, e, _, _ in spans] == ["Funda"]


def test_glued_tags_dropped_and_name_runs_merged():
    from pusula.clean_real import GLUED_TAG, NAME_RUN

    assert GLUED_TAG.search("Ç<AD>ŞAN personel") and GLUED_TAG.search("Ş<KURUM> yönetimi")
    assert not GLUED_TAG.search("<AD>'ye söyledim, <KURUM> aradı")
    assert NAME_RUN.sub("<AD>", "Dr. <AD> Akkaya ile <AD> <AD>'e yazdım") == "Dr. <AD> ile <AD>'e yazdım"
    assert NAME_RUN.sub("<AD>", "<AD> Hanım geldi") == "<AD> Hanım geldi"


def test_leftover_address_and_title():
    assert has_leftover_risk("Atatürk Cad. No: 5")
    assert has_leftover_risk("Çiçek Sokak 12'de oturuyorum")
    assert has_leftover_risk("doç dr Funda Kurt ilgisiz")
    assert not has_leftover_risk("İstanbul Pendik'te arıza var")
    assert not has_leftover_risk("Kadıköy şubesinde bekledim")
    assert not has_leftover_risk("sokakta kaldık")
    assert has_leftover_risk("Dr. Ayşe Yılmaz çok iyiydi")       # NER missed the name: drop
    assert has_leftover_risk("Müdire Pınar yalan söyledi")
    assert not has_leftover_risk("Dr. <AD> çok iyiydi")             # name already masked by NER
    assert not has_leftover_risk("doktor çok ilgisizdi")
    assert not has_leftover_risk("Müdür çok kaba, pilav Güzeldi, hava Soğuk")


def _fake_finder(texts: list[str]):
    out = []
    for text in texts:
        spans = []
        if "NERNAME" in text:
            i = text.index("NERNAME")
            spans.append((i, i + 7, "PERSON", 0.95))
        out.append(spans)
    return out


def _synthetic_csv(path: Path, n: int = 30) -> None:
    rows = []
    templates = [
        ("Servis çok yavaş kargo gecikti", "web"),
        ("Ürün bozuk geldi iade istiyorum", "forum"),
        ("Personel ilgisizdi bekledim", "maps"),
        ("<AD> <TARIH> randevu aldı", "web"),
        ("!!!", "forum"),
        ("Atatürk Cad. No: 5 adresim", "maps"),
        ("çok kötü kokuyor", "web"),
        ("Çok kötü!", "forum"),
        ("Kadıköy şubesinde kaldı paket", "maps"),
    ]
    for i in range(n):
        body, kanal = templates[i % len(templates)]
        if i == 5:
            body = "bÃ¶lÃ¼mÃ¼nde sorun var müdür aradı"
        rows.append({"metin": body, "kanal": kanal, "kategori": f"k{i % 3}"})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metin", "kanal", "kategori"])
        w.writeheader()
        w.writerows(rows)


def test_cli_end_to_end(tmp_path: Path):
    inp = tmp_path / "raw.csv"
    _synthetic_csv(inp, 30)
    out1 = tmp_path / "out1"
    main([
        "--inp", str(inp), "--text-field", "metin", "--out-dir", str(out1),
        "--eval-n", "5", "--strata", "kanal", "--id-prefix", "t", "--seed", "42", "--no-ner",
    ])
    for name in ("train.jsonl", "eval.jsonl", "dropped.jsonl", "review.tsv", "report.json"):
        p = out1 / name
        assert p.is_file()
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert stat.S_IMODE(out1.stat().st_mode) == 0o700
    report = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
    assert report["input"] == 30
    assert report["ner_enabled"] is False
    train = [json.loads(l) for l in (out1 / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    eval_rows = [json.loads(l) for l in (out1 / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert len(train) + len(eval_rows) == report["clean"]
    assert len(eval_rows) <= min(5, report["clean"] // 5)
    t_ids = {r["id"] for r in train}
    e_ids = {r["id"] for r in eval_rows}
    assert not t_ids & e_ids
    t_keys = {dedupe_key(r["state"]) for r in train}
    e_keys = {dedupe_key(r["state"]) for r in eval_rows}
    assert not t_keys & e_keys

    out2 = tmp_path / "out2"
    main([
        "--inp", str(inp), "--text-field", "metin", "--out-dir", str(out2),
        "--eval-n", "5", "--strata", "kanal", "--id-prefix", "t", "--seed", "42", "--no-ner",
    ])
    for name in ("train.jsonl", "eval.jsonl", "report.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()


def test_process_rows_with_fake_ner(tmp_path: Path):
    items = list(iter_rows(_write_mini_csv(tmp_path), "metin"))
    train, eval_rows, dropped, review, report = process_rows(
        items,
        seed=7,
        id_prefix="x",
        eval_n=2,
        strata="kanal",
        use_ner=True,
        name_finder=_fake_finder,
        ner_min_score=0.5,
    )
    assert report["ner_enabled"] is True
    states = [r["state"] for r in train + eval_rows]
    assert len(states) == 2 and all("NERNAME" not in t and not TAG_RE.search(t) for t in states)
    assert report["replacements"]["isim_ner"] == 1
    assert [d["reason"] for d in dropped] == ["anlamsiz"]


def _write_mini_csv(tmp_path: Path) -> Path:
    p = tmp_path / "mini.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metin", "kanal"])
        w.writeheader()
        w.writerow({"metin": "NERNAME ile görüştüm sorun devam", "kanal": "a"})
        w.writerow({"metin": "genel şikayet hizmet kötü", "kanal": "b"})
        w.writerow({"metin": "<BILINMEYEN>", "kanal": "a"})
    return p


def test_stratified_allocation():
    rows = [{"meta": {"k": "a"}}, {"meta": {"k": "a"}}, {"meta": {"k": "b"}}, {"meta": {"k": "b"}}, {"meta": {"k": "b"}}]
    idx = stratified_eval_indices(rows, eval_n=10, strata_field="k", seed=1)
    assert len(idx) == 1
