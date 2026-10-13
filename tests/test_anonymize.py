import json
import re

from pusula.anonymize import Anonymizer, fake_tckn, luhn_ok, main, needs_review, tckn_ok


def test_checksums():
    assert tckn_ok("10000000146") and not tckn_ok("12345678901")
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    import random
    assert all(tckn_ok(fake_tckn(random.Random(i))) for i in range(20))


def test_personal_data_is_replaced_with_fakes():
    a = Anonymizer(seed=1)
    text = ("Merhaba, adım Hasan Kara. TC 10000000146, tel 0532 123 45 67, e-posta hasan.kara@gmail.com, "
            "IBAN TR33 0006 1005 1978 6457 8413 26, kart 4111 1111 1111 1111, sipariş no 98765432, plaka 34 ABC 123. "
            "Ayrıca Mehmet Bey aradı. @hasankara yazdı, https://site.com/x bakın.")
    out = a(text)
    for leaked in ["Hasan Kara", "10000000146", "123 45 67", "hasan.kara@gmail.com", "TR33 0006", "4111 1111",
                   "98765432", "34 ABC 123", "@hasankara", "site.com", "Mehmet Bey"]:
        assert leaked not in out, leaked
    assert re.search(r"\b05\d{2} \d{3} \d{2} \d{2}\b", out) and "@ornek.com.tr" in out   # still reads as personal data
    assert {"tckn", "telefon", "eposta", "iban", "kart", "numara", "plaka", "isim", "kullanici_adi", "url"} <= set(a.counts)


def test_ordinary_numbers_and_text_stay():
    a = Anonymizer()
    text = "Kargom 3 gündür gelmedi, 250 TL ödedim, 2026 yılında aldım."
    assert a(text) == text and a.counts == {}


def test_review_flags_capitalized_words():
    assert "Kadıköy" in needs_review("Paketim dün Kadıköy şubesinde kaldı.")
    assert needs_review("Paketim dün şubede kaldı.") == []


def test_cli(tmp_path):
    inp = tmp_path / "raw.jsonl"
    inp.write_text(json.dumps({"text": "Benim adım Ali Veli, numaram 05551234567.", "id": 1}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    main(["--inp", str(inp), "--out", str(tmp_path / "anon.jsonl"), "--review", str(tmp_path / "r.tsv")])
    row = json.loads((tmp_path / "anon.jsonl").read_text(encoding="utf-8"))
    assert "Ali Veli" not in row["text"] and "05551234567" not in row["text"] and row["id"] == 1


def test_bey_hanim_keeps_preceding_word():
    out = Anonymizer(seed=2)("Ayrıca Mehmet Bey aradı.")
    assert out.startswith("Ayrıca ") and "Mehmet" not in out and out.endswith(" Bey aradı.")
