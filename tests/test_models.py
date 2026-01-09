"""
Unit tests for Vietnamese Emoji Suggestion System
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    preprocess_text,
    TextPreprocessor,
    remove_urls,
    remove_emojis,
    extract_emojis,
    replace_teencode,
    normalize_repeated_chars,
    TEENCODE_MAP
)
from src.models import (
    KeywordBaseline,
    EnsembleEmojiModel,
    EMOTION_EMOJI_MAP,
    EMOJI_DESCRIPTIONS
)
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k
)


# ============================================================================
# PREPROCESSING TESTS
# ============================================================================

class TestPreprocessing:
    """Test preprocessing functions."""
    
    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more"
        result = remove_urls(text)
        assert "https://example.com" not in result
        assert "Check out" in result
    
    def test_remove_emojis(self):
        """Test emoji removal."""
        text = "Hello 😊 World 🎉"
        result = remove_emojis(text)
        assert "😊" not in result
        assert "🎉" not in result
        assert "Hello" in result
    
    def test_extract_emojis(self):
        """Test emoji extraction."""
        text = "Happy 😊 day 🎉🥳"
        emojis = extract_emojis(text)
        assert "😊" in emojis[0] or len(emojis) >= 1
    
    def test_normalize_repeated_chars(self):
        """Test repeated character normalization."""
        text = "vuiiiiii quáááá"
        result = normalize_repeated_chars(text)
        assert "iii" not in result  # Should be reduced
    
    def test_teencode_replacement(self):
        """Test teencode/slang replacement."""
        text = "ko bít sao"
        result = replace_teencode(text)
        assert "không" in result or "biết" in result
    
    def test_full_preprocessing(self):
        """Test full preprocessing pipeline."""
        text = "Vuiii quáaaa 😊 https://example.com"
        result = preprocess_text(text)
        assert "http" not in result
        assert "😊" not in result
    
    def test_preprocessor_class(self):
        """Test TextPreprocessor class."""
        preprocessor = TextPreprocessor()
        text = "Check https://test.com và nói dc ko?"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert "https" not in result


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestKeywordBaseline:
    """Test KeywordBaseline model."""
    
    @pytest.fixture
    def model(self):
        return KeywordBaseline()
    
    def test_suggest_happy(self, model):
        """Test suggestion for happy text."""
        result = model.suggest("Chúc mừng bạn!")
        assert isinstance(result, list)
        assert len(result) <= 3
        assert any(e in ["😊", "🎉", "🥳", "👍"] for e in result)
    
    def test_suggest_sad(self, model):
        """Test suggestion for sad text."""
        result = model.suggest("Buồn quá")
        assert isinstance(result, list)
        assert any(e in ["😢", "😭", "💔", "😞"] for e in result)
    
    def test_suggest_angry(self, model):
        """Test suggestion for angry text."""
        result = model.suggest("Tức quá!")
        assert isinstance(result, list)
        assert any(e in ["😠", "💢", "😤", "😡"] for e in result)
    
    def test_suggest_unknown(self, model):
        """Test suggestion for text without keywords."""
        result = model.suggest("xyz abc 123")
        assert isinstance(result, list)
        assert len(result) == 3  # Should return default
    
    def test_get_matched_keywords(self, model):
        """Test keyword matching detection."""
        keywords = model.get_matched_keywords("Chúc mừng bạn vui quá!")
        assert isinstance(keywords, list)
        assert "chúc mừng" in keywords or "vui" in keywords


class TestEnsembleModel:
    """Test EnsembleEmojiModel."""
    
    @pytest.fixture
    def model(self):
        # Use lightweight model for testing
        return EnsembleEmojiModel(use_sentiment=False, use_semantic=False)
    
    def test_suggest_voting(self, model):
        """Test voting method."""
        result = model.suggest("Vui quá!", method="voting")
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_suggest_weighted(self, model):
        """Test weighted method."""
        result = model.suggest("Vui quá!", method="weighted")
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_suggest_with_details(self, model):
        """Test detailed suggestion."""
        result = model.suggest_with_details("Chúc mừng!", method="weighted")
        assert isinstance(result, dict)
        assert "final_suggestions" in result
        assert "keyword_suggestions" in result
        assert "matched_keywords" in result
    
    def test_empty_input(self, model):
        """Test with empty input."""
        result = model.suggest("")
        assert isinstance(result, list)


# ============================================================================
# EVALUATION TESTS
# ============================================================================

class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_precision_at_k_perfect(self):
        """Test precision@k with perfect predictions."""
        true = ["😊", "🎉", "🥳"]
        pred = ["😊", "🎉", "🥳"]
        result = precision_at_k(true, pred, k=3)
        assert result == 1.0
    
    def test_precision_at_k_partial(self):
        """Test precision@k with partial match."""
        true = ["😊", "🎉", "🥳"]
        pred = ["😊", "😄", "😁"]
        result = precision_at_k(true, pred, k=3)
        assert result == pytest.approx(1/3)
    
    def test_precision_at_k_none(self):
        """Test precision@k with no match."""
        true = ["😊", "🎉"]
        pred = ["😢", "😭", "💔"]
        result = precision_at_k(true, pred, k=3)
        assert result == 0.0
    
    def test_recall_at_k(self):
        """Test recall@k."""
        true = ["😊", "🎉", "🥳"]
        pred = ["😊", "🎉"]
        result = recall_at_k(true, pred, k=2)
        assert result == pytest.approx(2/3)
    
    def test_hit_rate_at_k_hit(self):
        """Test hit rate with a hit."""
        true = ["😊", "🎉"]
        pred = ["😄", "😊", "😁"]
        result = hit_rate_at_k(true, pred, k=3)
        assert result == 1.0
    
    def test_hit_rate_at_k_miss(self):
        """Test hit rate with no hit."""
        true = ["😊", "🎉"]
        pred = ["😢", "😭", "💔"]
        result = hit_rate_at_k(true, pred, k=3)
        assert result == 0.0
    
    def test_mrr_first(self):
        """Test MRR when correct answer is first."""
        true = ["😊"]
        pred = ["😊", "😄", "😁"]
        result = mrr(true, pred)
        assert result == 1.0
    
    def test_mrr_second(self):
        """Test MRR when correct answer is second."""
        true = ["😊"]
        pred = ["😄", "😊", "😁"]
        result = mrr(true, pred)
        assert result == 0.5
    
    def test_ndcg_at_k(self):
        """Test NDCG@k."""
        true = ["😊", "🎉"]
        pred = ["😊", "🎉", "😄"]
        result = ndcg_at_k(true, pred, k=3)
        assert result > 0.0


# ============================================================================
# DATA TESTS
# ============================================================================

class TestDataStructures:
    """Test data structures and constants."""
    
    def test_emotion_emoji_map(self):
        """Test EMOTION_EMOJI_MAP structure."""
        assert isinstance(EMOTION_EMOJI_MAP, dict)
        assert "joy" in EMOTION_EMOJI_MAP
        assert "sadness" in EMOTION_EMOJI_MAP
        assert len(EMOTION_EMOJI_MAP) >= 8
    
    def test_emoji_descriptions(self):
        """Test EMOJI_DESCRIPTIONS structure."""
        assert isinstance(EMOJI_DESCRIPTIONS, dict)
        assert "😊" in EMOJI_DESCRIPTIONS
        assert len(EMOJI_DESCRIPTIONS) >= 50
    
    def test_teencode_map(self):
        """Test TEENCODE_MAP structure."""
        assert isinstance(TEENCODE_MAP, dict)
        assert "ko" in TEENCODE_MAP
        assert TEENCODE_MAP["ko"] == "không"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test full suggestion pipeline."""
        from src.models import EnsembleEmojiModel
        from src.preprocessing import preprocess_text
        
        text = "Chúc mừng bạn đậu tuyển dụng! 🎉"
        
        # Preprocess
        processed = preprocess_text(text)
        assert "🎉" not in processed
        
        # Suggest
        model = EnsembleEmojiModel(use_sentiment=False, use_semantic=False)
        suggestions = model.suggest(text)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1
    
    def test_batch_processing(self):
        """Test batch processing."""
        from src.models import EnsembleEmojiModel
        
        texts = [
            "Vui quá!",
            "Buồn quá",
            "Tức ghê!"
        ]
        
        model = EnsembleEmojiModel(use_sentiment=False, use_semantic=False)
        
        for text in texts:
            result = model.suggest(text)
            assert isinstance(result, list)
            assert len(result) >= 1


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
