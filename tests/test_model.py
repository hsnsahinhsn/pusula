"""
Unit tests for Pusula architecture, DecisionHead, and candidate masking.
"""
import unittest

try:
    import torch
    from model import DecisionHead
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPusulaModel(unittest.TestCase):

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for model tests")
    def test_decision_head_shapes(self):
        hidden_size = 768
        num_total_options = 10
        batch_size = 4
        
        head = DecisionHead(hidden_size=hidden_size, num_total_options=num_total_options, dropout=0.1)
        dummy_pooled = torch.randn(batch_size, hidden_size)
        
        choice_logits, score_preds, action_logits = head(dummy_pooled)
        
        self.assertEqual(choice_logits.shape, (batch_size, num_total_options))
        self.assertEqual(score_preds.shape, (batch_size,))
        self.assertEqual(action_logits.shape, (batch_size,))
        
        # Scores must be bounded [0.0, 1.0] via Sigmoid
        self.assertTrue(torch.all(score_preds >= 0.0))
        self.assertTrue(torch.all(score_preds <= 1.0))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for model tests")
    def test_dynamic_candidate_masking_logic(self):
        """Verify that applying candidate masks sets excluded choices to -inf or zero probability."""
        batch_size = 2
        num_options = 5
        
        raw_logits = torch.tensor([
            [2.0, 5.0, 1.0, 0.5, -1.0],
            [1.0, 2.0, 8.0, 3.0, 0.0]
        ])
        
        candidate_mask = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0]
        ])
        
        masked_logits = raw_logits.clone()
        masked_logits = masked_logits.masked_fill(candidate_mask == 0, -1e4)
        probs = torch.softmax(masked_logits, dim=-1)
        
        # Excluded indices must have zero probability
        self.assertLess(probs[0, 1].item(), 1e-4)
        self.assertLess(probs[0, 3].item(), 1e-4)
        self.assertLess(probs[0, 4].item(), 1e-4)
        self.assertLess(probs[1, 0].item(), 1e-4)
        self.assertLess(probs[1, 2].item(), 1e-4)
        self.assertLess(probs[1, 4].item(), 1e-4)
        
        # Probabilities of allowed options must sum to ~1.0
        self.assertAlmostEqual(probs[0, 0].item() + probs[0, 2].item(), 1.0, places=4)
        self.assertAlmostEqual(probs[1, 1].item() + probs[1, 3].item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
