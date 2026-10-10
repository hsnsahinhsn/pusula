"""
Integration tests for Pusula FastAPI server endpoints.
"""
import unittest

try:
    from fastapi.testclient import TestClient
    from serve import app
    HAS_SERVER_DEPS = True
except ImportError:
    HAS_SERVER_DEPS = False


class TestPusulaServer(unittest.TestCase):

    def setUp(self):
        if HAS_SERVER_DEPS:
            self.client = TestClient(app)

    @unittest.skipUnless(HAS_SERVER_DEPS, "FastAPI and testclient are required for server tests")
    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "pusula-decision-engine")
        self.assertIn("version", data)

    @unittest.skipUnless(HAS_SERVER_DEPS, "FastAPI and testclient are required for server tests")
    def test_taxonomies_endpoint(self):
        response = self.client.get("/taxonomies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_domains"], 30)
        self.assertIn("corporate_accounting_and_tax", data["domains"])

    @unittest.skipUnless(HAS_SERVER_DEPS, "FastAPI and testclient are required for server tests")
    def test_decide_endpoint_contract(self):
        payload = {
            "context": "Kartımdan bilgim dışında para çekilmiş acil bloke edin.",
            "options": ["supheli_islem_ve_fraud", "kredi_faiz_ve_mevduat", "genel_bilgi"]
        }
        response = self.client.post("/decide", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("choice", data)
        self.assertIn("choice_confidence", data)
        self.assertIn("score", data)
        self.assertIn("action_required", data)
        self.assertIn("latency_ms", data)
        self.assertIsInstance(data["action_required"], bool)
        self.assertTrue(0.0 <= data["score"] <= 1.0)

    @unittest.skipUnless(HAS_SERVER_DEPS, "FastAPI and testclient are required for server tests")
    def test_batch_decide_endpoint(self):
        payload = {
            "items": [
                {
                    "context": "Yarın için şirket toplantı odasını ayırt.",
                    "options": ["takvim_ayarla", "e_posta_gonder"]
                },
                {
                    "context": "E-faturadaki tevkifat oranını kontrol et.",
                    "options": ["muhasebe_tevkifat", "kargo_takibi"]
                }
            ]
        }
        response = self.client.post("/batch-decide", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        self.assertIn("total_latency_ms", data)


if __name__ == "__main__":
    unittest.main()
