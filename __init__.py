"""
Turkish Decision Model (Türkçe Karar Modeli)
The ultra-fast, non-autoregressive, calibrated universal decision engine for Turkish AI workflows & agentic routing.
"""
from model import TurkishDecisionModel, DecisionHead
from config import TrainingConfig, GenerationConfig, InferenceConfig

__version__ = "1.0.0"
__all__ = [
    "TurkishDecisionModel",
    "DecisionHead",
    "TrainingConfig",
    "GenerationConfig",
    "InferenceConfig"
]
