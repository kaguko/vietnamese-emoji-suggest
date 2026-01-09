"""
Vietnamese Emoji Suggestion System - Source Package
"""

from src.preprocessing import preprocess_text, TextPreprocessor
from src.models import (
    KeywordBaseline,
    SentimentEmojisModel,
    SemanticMatchingModel,
    EnsembleEmojiModel,
    EMOTION_EMOJI_MAP,
    EMOJI_DESCRIPTIONS,
)
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    evaluate_model,
    compare_models,
    error_analysis,
)

__version__ = "1.0.0"
__all__ = [
    # Preprocessing
    "preprocess_text",
    "TextPreprocessor",
    # Models
    "KeywordBaseline",
    "SentimentEmojisModel",
    "SemanticMatchingModel",
    "EnsembleEmojiModel",
    # Data
    "EMOTION_EMOJI_MAP",
    "EMOJI_DESCRIPTIONS",
    # Evaluation
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "evaluate_model",
    "compare_models",
    "error_analysis",
]
