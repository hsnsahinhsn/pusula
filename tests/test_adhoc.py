from pusula.adhoc import PROMPT, PROMPT_WIDE, SCHEMA, SCHEMA_WIDE, to_questions


def test_wide_mode_asks_for_long_option_lists():
    assert PROMPT_WIDE != PROMPT and "25–60" in PROMPT_WIDE and "{state}" in PROMPT_WIDE
    items = lambda s: s["properties"]["sorular"]["items"]["properties"]["secenekler"]["maxItems"]  # noqa: E731
    assert items(SCHEMA) == 30 and items(SCHEMA_WIDE) == 60
    out = {"sorular": [{"talimat": "Hangi kalem?", "tip": "choice",
                        "secenekler": [{"anahtar": f"kalem {i}", "aciklama": ""} for i in range(55)]}]}
    assert len(to_questions(out)["adhoc_1"].option_keys()) == 55
