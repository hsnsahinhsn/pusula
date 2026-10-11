import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pusula.schema import answer_from_probs  # noqa: E402
from pusula.serve import create_app  # noqa: E402


class FakeEngine:
    model_name = "fake"

    def decide_many(self, requests):
        out = []
        for _, qs in requests:
            out.append({n: answer_from_probs(q, [1.0] + [0.0] * (len(q.option_keys()) - 1))
                        for n, q in qs.items()})
        return out

    def decide(self, state, questions):
        return self.decide_many([(state, questions)])[0]

    def route(self, state, universal=True):
        return {"path": "genel_sohbet.sohbet.selamlasma", "answers": {}}


BODY = {
    "state": "Kartım çalındı, hemen bloke edin!",
    "questions": {
        "ekip": {"type": "choice", "instructions": "Hangi ekip?",
                 "criteria": {"kart": "Kart işlemleri", "kredi": "Kredi"}},
        "acil": {"type": "noul", "instructions": "Acil mi?"},
    },
}


@pytest.fixture
def client():
    return TestClient(create_app(engine=FakeEngine()))


def test_health_and_ready(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/ready").status_code == 200


def test_not_ready_without_model(monkeypatch):
    monkeypatch.delenv("PUSULA_MODEL_DIR", raising=False)
    c = TestClient(create_app())
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.post("/v1/systemone", json=BODY).status_code == 503


def test_systemone_contract(client):
    r = client.post("/v1/systemone", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["answers"]["ekip"]["choice"] == "kart"
    assert data["answers"]["acil"]["type"] == "noul"
    assert data["latency_ms"] >= 0


def test_batch_and_route(client):
    r = client.post("/v1/systemone/batch", json={"requests": [BODY, BODY]})
    assert r.status_code == 200 and len(r.json()["results"]) == 2
    assert client.post("/v1/route", json={"state": "merhaba"}).json()["path"].count(".") == 2


@pytest.mark.parametrize("bad", [
    {"state": "x", "questions": {}},
    {"state": "", "questions": BODY["questions"]},
    {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"tek": None}}}},
    {"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}},
])
def test_validation_errors(client, bad):
    assert client.post("/v1/systemone", json=bad).status_code == 422


def test_taxonomy_endpoint(client):
    data = client.get("/v1/taxonomy").json()
    assert "domains" in data and "universal" in data
