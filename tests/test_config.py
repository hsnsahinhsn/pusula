"""
Unit tests for Pusula configuration and taxonomies.
"""
import unittest
from config import GenerationConfig, TrainingConfig, InferenceConfig

class TestPusulaConfig(unittest.TestCase):
    def test_generation_config_taxonomies(self):
        cfg = GenerationConfig()
        taxonomies = cfg.taxonomies
        self.assertGreaterEqual(len(taxonomies), 30, "Taxonomy count must be at least 30 universal macro-domains.")
        
        # Verify corporate accounting and finance domains exist
        self.assertIn("corporate_accounting_and_tax", taxonomies)
        self.assertIn("banking_finance_crypto", taxonomies)
        self.assertIn("global_master_routing", taxonomies)
        
        # Validate each taxonomy structure
        for key, data in taxonomies.items():
            self.assertIn("description", data, f"Domain {key} is missing 'description'")
            self.assertIn("options", data, f"Domain {key} is missing 'options'")
            self.assertIsInstance(data["options"], list)
            self.assertGreaterEqual(len(data["options"]), 2, f"Domain {key} must have at least 2 candidate options")
            # Check uniqueness of options
            self.assertEqual(len(data["options"]), len(set(data["options"])), f"Duplicate options found in {key}")

    def test_training_config_defaults(self):
        t_cfg = TrainingConfig()
        self.assertEqual(t_cfg.base_model_name, "convaiinnovations/laya-multilingual")
        self.assertEqual(t_cfg.mixed_precision, "bf16")
        self.assertEqual(t_cfg.loss_weight_choice, 1.0)
        self.assertEqual(t_cfg.loss_weight_score, 0.5)
        self.assertEqual(t_cfg.loss_weight_action, 0.5)
        self.assertEqual(t_cfg.max_length, 512)

    def test_inference_config_defaults(self):
        i_cfg = InferenceConfig()
        self.assertEqual(i_cfg.port, 8080)
        self.assertEqual(i_cfg.host, "0.0.0.0")

if __name__ == "__main__":
    unittest.main()
