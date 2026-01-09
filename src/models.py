"""
Model implementations for Vietnamese Emoji Suggestion System.

This module contains:
- KeywordBaseline: Rule-based keyword matching
- SentimentEmojisModel: Emotion detection using BERT
- SemanticMatchingModel: Semantic similarity for emoji matching
- EnsembleEmojiModel: Combined ensemble of all approaches
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np

# Import preprocessing
from src.preprocessing import preprocess_text, TextPreprocessor


# ============================================================================
# EMOJI DATABASE
# ============================================================================

# Comprehensive emoji descriptions for semantic matching
EMOJI_DESCRIPTIONS = {
    # Joy / Happy
    "😊": "nụ cười vui vẻ hạnh phúc ấm áp thân thiện",
    "😄": "cười tươi vui vẻ hạnh phúc sung sướng",
    "😁": "cười toe toét vui vẻ hào hứng",
    "😃": "cười lớn vui vẻ phấn khích",
    "🥳": "tiệc tùng ăn mừng vui vẻ lễ hội sinh nhật",
    "🎉": "bữa tiệc ăn mừng chúc mừng thành công",
    "🎊": "confetti ăn mừng tiệc tùng vui vẻ",
    "✨": "lấp lánh tuyệt vời magical đặc biệt",
    "🌟": "ngôi sao sáng xuất sắc tuyệt vời",
    "💫": "chóng mặt ngôi sao kỳ diệu tuyệt vời",
    "🤩": "mắt sao ngưỡng mộ phấn khích tuyệt vời",
    "😆": "cười híp mắt vui vẻ hài hước",
    "😂": "cười chảy nước mắt buồn cười hài hước",
    "🤣": "lăn ra cười buồn cười hài hước",
    "😌": "nhẹ nhõm bình yên thoải mái hài lòng",
    "🙂": "cười nhẹ bình thường ok ổn",
    "😏": "cười nửa miệng tự tin ranh mãnh",
    
    # Love / Affection
    "❤️": "trái tim yêu thương tình cảm",
    "💕": "hai trái tim yêu thương lãng mạn",
    "💖": "trái tim lấp lánh yêu thương đặc biệt",
    "💗": "trái tim đập yêu thương rung động",
    "💓": "trái tim đập yêu thương hồi hộp",
    "💞": "trái tim xoay yêu thương say đắm",
    "💘": "trái tim tên Cupid yêu thương tình yêu",
    "😍": "mắt trái tim yêu thương ngưỡng mộ si mê",
    "🥰": "yêu thương hạnh phúc trái tim",
    "😘": "hôn gió yêu thương tình cảm",
    "😗": "hôn môi yêu thương tình cảm",
    "😙": "hôn mắt nhắm yêu thương",
    "😚": "hôn má yêu thương",
    "🤗": "ôm thân thiện yêu thương chào đón",
    
    # Sadness
    "😢": "khóc buồn đau lòng thất vọng",
    "😭": "khóc nức nở buồn đau khổ",
    "😞": "thất vọng buồn chán nản",
    "😔": "buồn bã suy nghĩ thất vọng",
    "🥺": "xin xỏ buồn thương cảm động",
    "😿": "mèo khóc buồn đáng thương",
    "💔": "trái tim vỡ đau lòng chia tay thất vọng",
    "😥": "lo lắng buồn thất vọng",
    "😰": "lo lắng căng thẳng sợ hãi",
    "☹️": "mặt buồn không vui thất vọng",
    "😩": "mệt mỏi chán nản kiệt sức",
    "😫": "mệt mỏi chán ngán kiệt sức",
    "😖": "khó chịu đau đớn bực bội",
    
    # Anger
    "😠": "giận tức bực mình khó chịu",
    "😡": "giận dữ tức giận đỏ mặt",
    "🤬": "chửi thề giận dữ tức điên",
    "😤": "hậm hực tức giận bực bội",
    "💢": "tức giận nổi giận bùng nổ",
    "👿": "quỷ giận dữ ác độc",
    "👊": "đấm đánh mạnh mẽ giận",
    "🔥": "lửa nóng giận dữ mạnh mẽ",
    "😒": "không hài lòng khó chịu chán",
    "🙄": "đảo mắt chán ngán không tin",
    
    # Fear / Worry
    "😨": "sợ hãi kinh hãi hoảng",
    "😱": "kinh hoàng sợ hãi shock",
    "😰": "lo lắng căng thẳng sợ",
    "😟": "lo lắng buồn bã sợ",
    "😬": "căng thẳng ngại ngùng khó xử",
    "🥶": "lạnh run rẩy sợ hãi",
    "😵": "chóng mặt choáng váng shock",
    "🙀": "mèo sợ kinh hoàng hoảng",
    "💀": "đầu lâu chết kinh khủng",
    "👻": "ma sợ hãi halloween",
    "💦": "mồ hôi lo lắng căng thẳng",
    
    # Surprise
    "😮": "ngạc nhiên ồ wow",
    "😲": "sốc ngạc nhiên kinh ngạc",
    "🤯": "bùng nổ đầu sốc kinh ngạc",
    "😯": "im lặng ngạc nhiên",
    "🙊": "khỉ che miệng ngạc nhiên im lặng",
    "❓": "hỏi thắc mắc không hiểu",
    "❗": "chú ý quan trọng ngạc nhiên",
    "⁉️": "hỏi ngạc nhiên sốc",
    "😳": "ngượng ngùng bất ngờ đỏ mặt",
    
    # Disgust
    "🤢": "buồn nôn ghê tởm kinh",
    "🤮": "nôn ói ghê tởm",
    "😖": "khó chịu đau đớn ghê",
    "😷": "đeo khẩu trang bệnh ghê",
    "🚫": "cấm không được không",
    "❌": "sai không không được",
    "👎": "không thích tệ dở",
    
    # Trust / Support
    "🤝": "bắt tay hợp tác tin tưởng",
    "👍": "tốt hay đồng ý ủng hộ",
    "💪": "mạnh mẽ cố gắng ủng hộ",
    "✅": "đúng hoàn thành xong",
    "💯": "hoàn hảo tuyệt vời 100%",
    "👏": "vỗ tay khen ngợi giỏi",
    "🙏": "cảm ơn cầu nguyện xin",
    "👌": "ok tốt được đồng ý",
    "✌️": "hòa bình chiến thắng ok",
    "🌈": "cầu vồng hy vọng đẹp",
    
    # Anticipation / Excitement
    "🤞": "chéo ngón hy vọng mong",
    "⏰": "đồng hồ thời gian chờ đợi",
    "⏳": "cát rơi chờ đợi thời gian",
    "🎂": "bánh sinh nhật tiệc mừng",
    "🎁": "quà tặng bất ngờ",
    "🏖️": "bãi biển nghỉ hè thư giãn",
    "☀️": "mặt trời nắng vui vẻ",
    "⭐": "ngôi sao đánh giá tốt",
    
    # Thinking / Confusion
    "🤔": "suy nghĩ thắc mắc cân nhắc",
    "🧐": "kiểm tra xem xét tò mò",
    "😕": "bối rối không hiểu thắc mắc",
    "😑": "bình thường không biểu cảm chán",
    "😐": "trung tính bình thường không cảm xúc",
    "🙃": "đảo ngược mỉa mai hài hước",
    
    # Other emotions
    "😅": "ngại ngùng hài hước lo lắng nhẹ",
    "😇": "thiên thần ngoan tốt bụng",
    "🤡": "hề hài hước ngốc nghếch",
    "😎": "cool ngầu tự tin",
    "🥴": "say xỉn chóng mặt",
    "🤪": "điên crazy vui nhộn",
    "😜": "nháy mắt lè lưỡi nghịch ngợm",
    "😝": "lè lưỡi nghịch vui",
    "🤭": "che miệng cười ngại ngùng",
    "🥲": "cười mà muốn khóc xúc động",
}

# Emotion to emoji mapping (based on voting)
EMOTION_EMOJI_MAP = {
    "joy": ["😊", "🎉", "😄", "🥳", "✨", "🤩", "😁", "🌟"],
    "sadness": ["😢", "😭", "💔", "😞", "😔", "🥺", "☹️", "😿"],
    "anger": ["😠", "💢", "😤", "😡", "🤬", "👿", "🔥", "👊"],
    "fear": ["😨", "😱", "😰", "😟", "😬", "💀", "👻", "💦"],
    "surprise": ["😮", "😲", "🤯", "😯", "❓", "😳", "🙊", "❗"],
    "disgust": ["🤢", "🤮", "😖", "😷", "👎", "❌", "🚫", "😒"],
    "trust": ["🤝", "💪", "👍", "✅", "💯", "👏", "🙏", "👌"],
    "anticipation": ["🤞", "⏰", "🎉", "✨", "🎂", "😊", "🎁", "⏳"],
}


# ============================================================================
# BASELINE MODELS
# ============================================================================

class KeywordBaseline:
    """
    Rule-based emoji suggestion using keyword matching.
    
    This is the simplest baseline that maps keywords to emojis.
    Expected accuracy: ~45%
    """
    
    def __init__(self):
        self.keyword_emoji_map = {
            # Joy keywords
            "chúc mừng": ["😊", "🎉", "🥳"],
            "vui": ["😊", "😄", "🎉"],
            "tuyệt vời": ["🤩", "✨", "👏"],
            "tuyệt": ["👍", "✨", "🌟"],
            "hay": ["👍", "🔥", "✨"],
            "yêu": ["❤️", "💕", "😍"],
            "thích": ["❤️", "👍", "😊"],
            "cảm ơn": ["🙏", "❤️", "😊"],
            "hạnh phúc": ["😊", "🥰", "💕"],
            "giỏi": ["👏", "💪", "🌟"],
            "xuất sắc": ["🏆", "👏", "✨"],
            "tốt": ["👍", "😊", "✅"],
            "ok": ["👍", "👌", "✅"],
            "được": ["👍", "👌", "😊"],
            "thành công": ["🎉", "🏆", "✨"],
            "chiến thắng": ["🏆", "🎉", "💪"],
            "ăn mừng": ["🎉", "🥳", "🍻"],
            "sinh nhật": ["🎂", "🎉", "🥳"],
            
            # Sadness keywords
            "buồn": ["😢", "😭", "💔"],
            "nhớ": ["🥺", "😢", "💔"],
            "đau": ["💔", "😢", "😞"],
            "khổ": ["😭", "💔", "😢"],
            "thất vọng": ["😞", "😔", "💔"],
            "chán": ["😒", "😔", "😕"],
            "mệt": ["😩", "😔", "💤"],
            "cô đơn": ["😢", "🥺", "💔"],
            "thương": ["🥺", "💔", "😢"],
            "tiếc": ["😔", "😞", "💔"],
            "chia tay": ["💔", "😢", "😭"],
            "mất": ["😢", "💔", "😞"],
            
            # Anger keywords
            "giận": ["😠", "💢", "😤"],
            "tức": ["😤", "💢", "😠"],
            "bực": ["😤", "😒", "💢"],
            "ghét": ["😠", "👎", "💢"],
            "điên": ["🤬", "💢", "😡"],
            "quá đáng": ["😠", "💢", "👎"],
            "sốt ruột": ["😤", "⏰", "😠"],
            "khó chịu": ["😒", "😤", "💢"],
            
            # Fear keywords
            "sợ": ["😨", "😱", "😰"],
            "lo": ["😰", "😟", "😥"],
            "căng thẳng": ["😰", "💦", "😬"],
            "hoang mang": ["😰", "😟", "❓"],
            "run": ["😨", "😱", "💦"],
            "hồi hộp": ["😬", "💓", "😰"],
            "thi": ["😰", "📚", "🤞"],
            "đáng sợ": ["😱", "😨", "👻"],
            
            # Surprise keywords
            "ngạc nhiên": ["😮", "😲", "🤯"],
            "bất ngờ": ["🤯", "😮", "🎉"],
            "sốc": ["😱", "🤯", "😲"],
            "không ngờ": ["😲", "😮", "🤯"],
            "wow": ["🤩", "😮", "✨"],
            "trời ơi": ["😱", "😮", "🙀"],
            "ủa": ["🤔", "😮", "❓"],
            "thật sao": ["😲", "😮", "❓"],
            
            # Disgust keywords
            "ghê": ["🤢", "🤮", "😖"],
            "kinh": ["😱", "🤢", "😖"],
            "dơ": ["🤢", "😷", "🚫"],
            "bẩn": ["🤢", "😷", "🚫"],
            "tệ": ["👎", "😤", "💔"],
            "dở": ["👎", "😒", "😕"],
            
            # Trust keywords
            "tin": ["🤝", "💪", "👍"],
            "ủng hộ": ["👍", "💪", "🤝"],
            "yên tâm": ["😌", "🤗", "👌"],
            "chắc chắn": ["✅", "💯", "👍"],
            "cố gắng": ["💪", "✨", "🔥"],
            "cố lên": ["💪", "✨", "🌟"],
            
            # Anticipation keywords
            "mong": ["🤞", "😊", "✨"],
            "chờ": ["⏰", "🤞", "😊"],
            "háo hức": ["🤩", "😆", "🎊"],
            "hy vọng": ["🤞", "✨", "🙏"],
            "sắp": ["🎉", "⏰", "✨"],
            "nghỉ": ["🏖️", "😌", "🎉"],
            
            # Common expressions
            "haha": ["😂", "🤣", "😆"],
            "hehe": ["😄", "😊", "😆"],
            "hihi": ["😊", "🙈", "😄"],
            "huhu": ["😢", "😭", "🥺"],
            "ơ": ["🤔", "😮", "❓"],
            "à": ["😊", "👌", "🤔"],
        }
        
        # Preprocessor for text cleaning
        self.preprocessor = TextPreprocessor(remove_emoji=True)
    
    def suggest(self, text: str) -> List[str]:
        """
        Suggest emojis based on keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            List of up to 3 suggested emojis
        """
        # Preprocess text
        text_clean = self.preprocessor.preprocess(text)
        
        suggestions = []
        matched_keywords = []
        
        # Check for keyword matches
        for keyword, emojis in self.keyword_emoji_map.items():
            if keyword in text_clean:
                matched_keywords.append(keyword)
                suggestions.extend(emojis)
        
        if not suggestions:
            # Default fallback
            return ["🤔", "😊", "👍"]
        
        # Count and deduplicate
        emoji_counts = Counter(suggestions)
        top_emojis = [emoji for emoji, _ in emoji_counts.most_common(3)]
        
        return top_emojis
    
    def get_matched_keywords(self, text: str) -> List[str]:
        """Get list of matched keywords for debugging."""
        text_clean = self.preprocessor.preprocess(text)
        matched = []
        for keyword in self.keyword_emoji_map:
            if keyword in text_clean:
                matched.append(keyword)
        return matched


class SentimentEmojisModel:
    """
    Emotion detection using pre-trained sentiment model.
    
    Uses a multilingual BERT model for sentiment/emotion classification,
    then maps detected emotion to appropriate emojis.
    Expected accuracy: ~55%
    """
    
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        
        # Preprocessor
        self.preprocessor = TextPreprocessor(remove_emoji=True)
        
        # Sentiment to emotion mapping (5-star to 8 emotions)
        # 1-2 stars: negative emotions
        # 3 stars: neutral
        # 4-5 stars: positive emotions
        self.sentiment_emotion_map = {
            1: "anger",      # Very negative
            2: "sadness",    # Negative
            3: "trust",      # Neutral (default to trust)
            4: "joy",        # Positive
            5: "joy",        # Very positive
        }
        
        # Emotion to emoji mapping
        self.emotion_emoji_map = EMOTION_EMOJI_MAP
    
    def _load_model(self):
        """Lazy load the model to save memory."""
        if self._is_loaded:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self._is_loaded = True
            print(f"Loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to random emotion assignment.")
    
    def predict_emotion(self, text: str) -> Tuple[str, float]:
        """
        Predict emotion from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (emotion_label, confidence)
        """
        self._load_model()
        
        if not self._is_loaded:
            # Fallback: random emotion
            import random
            emotion = random.choice(list(self.emotion_emoji_map.keys()))
            return emotion, 0.5
        
        import torch
        
        # Preprocess
        text_clean = self.preprocessor.preprocess(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            sentiment_idx = torch.argmax(probs, dim=1).item() + 1  # 1-5
            confidence = probs[0, sentiment_idx - 1].item()
        
        # Map to emotion
        emotion = self.sentiment_emotion_map.get(sentiment_idx, "trust")
        
        return emotion, confidence
    
    def suggest(self, text: str) -> List[str]:
        """
        Suggest emojis based on detected emotion.
        
        Args:
            text: Input text
            
        Returns:
            List of up to 3 suggested emojis
        """
        emotion, confidence = self.predict_emotion(text)
        emojis = self.emotion_emoji_map.get(emotion, ["😊", "👍", "✨"])
        return emojis[:3]


class SemanticMatchingModel:
    """
    Semantic similarity-based emoji suggestion.
    
    Uses sentence embeddings to find emojis whose descriptions
    are semantically similar to the input text.
    Expected accuracy: ~50%
    """
    
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        self.model_name = model_name
        self.model = None
        self._is_loaded = False
        
        # Preprocessor
        self.preprocessor = TextPreprocessor(remove_emoji=True)
        
        # Emoji descriptions
        self.emoji_descriptions = EMOJI_DESCRIPTIONS
        
        # Pre-computed emoji embeddings
        self.emoji_embeddings = {}
        self.emoji_list = []
    
    def _load_model(self):
        """Lazy load the model and compute emoji embeddings."""
        if self._is_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.model_name)
            
            # Compute emoji embeddings
            print("Computing emoji embeddings...")
            for emoji, description in self.emoji_descriptions.items():
                embedding = self.model.encode(description)
                self.emoji_embeddings[emoji] = embedding
                self.emoji_list.append(emoji)
            
            self._is_loaded = True
            print(f"Loaded model: {self.model_name}")
            print(f"Computed embeddings for {len(self.emoji_list)} emojis")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to random emoji selection.")
    
    def suggest(self, text: str, top_k: int = 3) -> List[str]:
        """
        Suggest emojis based on semantic similarity.
        
        Args:
            text: Input text
            top_k: Number of suggestions to return
            
        Returns:
            List of suggested emojis
        """
        self._load_model()
        
        if not self._is_loaded:
            # Fallback: random emojis
            import random
            return random.sample(self.emoji_list or ["😊", "👍", "❤️"], min(top_k, 3))
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Preprocess
        text_clean = self.preprocessor.preprocess(text)
        
        # Encode input text
        text_embedding = self.model.encode(text_clean)
        
        # Compute similarities
        similarities = []
        for emoji, emoji_embed in self.emoji_embeddings.items():
            sim = cosine_similarity([text_embedding], [emoji_embed])[0][0]
            similarities.append((emoji, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [emoji for emoji, _ in similarities[:top_k]]
    
    def get_similarity_scores(self, text: str) -> List[Tuple[str, float]]:
        """Get all emoji similarity scores for debugging."""
        self._load_model()
        
        if not self._is_loaded:
            return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        text_clean = self.preprocessor.preprocess(text)
        text_embedding = self.model.encode(text_clean)
        
        similarities = []
        for emoji, emoji_embed in self.emoji_embeddings.items():
            sim = cosine_similarity([text_embedding], [emoji_embed])[0][0]
            similarities.append((emoji, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleEmojiModel:
    """
    Ensemble model combining multiple approaches.
    
    Combines:
    - Keyword matching (25%)
    - Sentiment analysis (35%)
    - Semantic matching (40%)
    
    Expected accuracy: ~60-62%
    """
    
    def __init__(
        self,
        use_sentiment: bool = True,
        use_semantic: bool = True,
        keyword_weight: float = 0.25,
        sentiment_weight: float = 0.35,
        semantic_weight: float = 0.40
    ):
        self.keyword_model = KeywordBaseline()
        self.sentiment_model = SentimentEmojisModel() if use_sentiment else None
        self.semantic_model = SemanticMatchingModel() if use_semantic else None
        
        self.keyword_weight = keyword_weight
        self.sentiment_weight = sentiment_weight if use_sentiment else 0
        self.semantic_weight = semantic_weight if use_semantic else 0
        
        # Normalize weights
        total_weight = self.keyword_weight + self.sentiment_weight + self.semantic_weight
        self.keyword_weight /= total_weight
        self.sentiment_weight /= total_weight
        self.semantic_weight /= total_weight
    
    def suggest(self, text: str, method: str = "weighted") -> List[str]:
        """
        Suggest emojis using ensemble of models.
        
        Args:
            text: Input text
            method: 'voting' for majority voting, 'weighted' for weighted combination
            
        Returns:
            List of up to 3 suggested emojis
        """
        # Get suggestions from all models
        keyword_result = self.keyword_model.suggest(text)
        sentiment_result = self.sentiment_model.suggest(text) if self.sentiment_model else []
        semantic_result = self.semantic_model.suggest(text) if self.semantic_model else []
        
        if method == "voting":
            return self._voting_ensemble(keyword_result, sentiment_result, semantic_result)
        else:
            return self._weighted_ensemble(keyword_result, sentiment_result, semantic_result)
    
    def _voting_ensemble(
        self,
        keyword_result: List[str],
        sentiment_result: List[str],
        semantic_result: List[str]
    ) -> List[str]:
        """Simple majority voting."""
        all_suggestions = keyword_result + sentiment_result + semantic_result
        votes = Counter(all_suggestions)
        return [emoji for emoji, _ in votes.most_common(3)]
    
    def _weighted_ensemble(
        self,
        keyword_result: List[str],
        sentiment_result: List[str],
        semantic_result: List[str]
    ) -> List[str]:
        """Weighted combination of suggestions."""
        emoji_scores = {}
        
        # Add weighted scores
        for i, emoji in enumerate(keyword_result[:3]):
            weight = self.keyword_weight * (3 - i) / 3  # Position-weighted
            emoji_scores[emoji] = emoji_scores.get(emoji, 0) + weight
        
        for i, emoji in enumerate(sentiment_result[:3]):
            weight = self.sentiment_weight * (3 - i) / 3
            emoji_scores[emoji] = emoji_scores.get(emoji, 0) + weight
        
        for i, emoji in enumerate(semantic_result[:3]):
            weight = self.semantic_weight * (3 - i) / 3
            emoji_scores[emoji] = emoji_scores.get(emoji, 0) + weight
        
        # Sort by score
        sorted_emojis = sorted(emoji_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [emoji for emoji, _ in sorted_emojis[:3]]
    
    def suggest_with_details(self, text: str, method: str = "weighted") -> Dict:
        """
        Suggest emojis with detailed breakdown.
        
        Returns dict with suggestions from each model and final result.
        """
        keyword_result = self.keyword_model.suggest(text)
        sentiment_result = self.sentiment_model.suggest(text) if self.sentiment_model else []
        semantic_result = self.semantic_model.suggest(text) if self.semantic_model else []
        
        if method == "voting":
            final_result = self._voting_ensemble(keyword_result, sentiment_result, semantic_result)
        else:
            final_result = self._weighted_ensemble(keyword_result, sentiment_result, semantic_result)
        
        # Get emotion if sentiment model is available
        emotion = None
        if self.sentiment_model:
            emotion, _ = self.sentiment_model.predict_emotion(text)
        
        return {
            "text": text,
            "keyword_suggestions": keyword_result,
            "sentiment_suggestions": sentiment_result,
            "semantic_suggestions": semantic_result,
            "final_suggestions": final_result,
            "detected_emotion": emotion,
            "method": method,
            "matched_keywords": self.keyword_model.get_matched_keywords(text)
        }


# ============================================================================
# EVALUATION
# ============================================================================

def precision_at_k(true_labels: List[str], predictions: List[str], k: int = 3) -> float:
    """
    Calculate precision@k.
    
    Args:
        true_labels: List of correct emoji labels
        predictions: List of predicted emojis
        k: Number of predictions to consider
        
    Returns:
        Precision score (0-1)
    """
    predictions = predictions[:k]
    true_set = set(true_labels)
    pred_set = set(predictions)
    
    if not predictions:
        return 0.0
    
    correct = len(true_set & pred_set)
    return correct / len(predictions)


def evaluate_model(model, test_data: List[Dict], k: int = 3) -> Dict:
    """
    Evaluate a model on test data.
    
    Args:
        model: Model with .suggest() method
        test_data: List of dicts with 'text' and 'emoji_1', 'emoji_2', 'emoji_3'
        k: Number of predictions for precision@k
        
    Returns:
        Dict with evaluation metrics
    """
    precisions = []
    correct_at_1 = 0
    
    for sample in test_data:
        text = sample['text']
        true_emojis = [sample.get('emoji_1'), sample.get('emoji_2'), sample.get('emoji_3')]
        true_emojis = [e for e in true_emojis if e]  # Remove None
        
        predictions = model.suggest(text)
        
        # Precision@k
        prec = precision_at_k(true_emojis, predictions, k)
        precisions.append(prec)
        
        # Accuracy@1 (is first prediction correct?)
        if predictions and predictions[0] in true_emojis:
            correct_at_1 += 1
    
    return {
        'precision_at_k': np.mean(precisions),
        'accuracy_at_1': correct_at_1 / len(test_data) if test_data else 0,
        'num_samples': len(test_data)
    }


if __name__ == "__main__":
    # Quick test
    print("=== TESTING MODELS ===\n")
    
    test_texts = [
        "Chúc mừng bạn đậu tuyển dụng!",
        "Buồn quá",
        "Tức ghê!",
        "Sợ quá!",
        "Thật sao!",
    ]
    
    # Test keyword baseline
    print("--- Keyword Baseline ---")
    baseline = KeywordBaseline()
    for text in test_texts:
        result = baseline.suggest(text)
        print(f"'{text}' -> {result}")
    
    print("\n--- Ensemble Model (keyword-only for quick test) ---")
    ensemble = EnsembleEmojiModel(use_sentiment=False, use_semantic=False)
    for text in test_texts:
        result = ensemble.suggest(text)
        print(f"'{text}' -> {result}")
