import pytest

from pusula.schema import Example, Question
from pusula.taxonomy import load_taxonomy


class FakeTokenizer:
    """Character-level stand-in for a HF tokenizer (no downloads in tests)."""

    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    bos_token_id = eos_token_id = None

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[3 + ord(c) % 250 for c in t[:200]] for t in texts]}

    def save_pretrained(self, path):
        pass


@pytest.fixture(scope="session")
def tax():
    return load_taxonomy()


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def choice_q():
    return Question(type="choice", instructions="Hangi ekip?",
                    criteria={"iade": "İade ve değişim", "fatura": "Fatura ve ödeme", "teknik": None})


@pytest.fixture
def example(tax):
    path = next(iter(tax.intent_paths()))
    qs = tax.questions_for(path)
    gen = path.as_labels()
    gen.update({"aciliyet": "2", "eylem_gerekli": "true", "insana_devret": "false", "guvenlik": "guvenli",
                "duygu": "endiseli", "kisisel_veri": "false", "mesaj_turu": "talep", "eksik_bilgi": "false",
                "hakaret": "yok", "dolandiricilik": "true", "can_guvenligi": "yok", "yasal_risk": "false",
                "ayrilma_riski": "hayir", "satis_firsati": "false"})
    return Example(id="ex1", state="Kartımdan bilgim dışında para çekildi.", questions=qs,
                   meta={"path": str(path)}, labels={"generator": gen})


@pytest.fixture
def tiny_engine(tokenizer):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from pusula.inference import DecisionEngine
    from pusula.model import PusulaModel

    torch.manual_seed(0)
    cfg = transformers.ModernBertConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=512, pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1,
        sep_token_id=2, global_attn_every_n_layers=1, local_attention=64,
    )
    encoder = transformers.ModernBertModel(cfg)
    model = PusulaModel(encoder).eval()
    return DecisionEngine(model, tokenizer, max_length=128, device=torch.device("cpu"))
