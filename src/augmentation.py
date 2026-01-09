"""
Data augmentation and weak labeling module for Vietnamese Emoji Suggestion System.

This module provides:
- Synonym replacement augmentation
- Weak labeling using rule-based methods
- Data validation and quality control
"""

import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Vietnamese synonym dictionary for augmentation
VIETNAMESE_SYNONYMS = {
    # Joy synonyms
    "vui": ["hạnh phúc", "sung sướng", "phấn khởi", "hào hứng"],
    "hạnh phúc": ["vui", "sung sướng", "mãn nguyện"],
    "tuyệt vời": ["tuyệt", "hay", "xuất sắc", "tốt lắm"],
    "hay": ["tốt", "tuyệt", "giỏi", "đỉnh"],
    "thích": ["yêu thích", "ưa", "mê"],
    
    # Sadness synonyms
    "buồn": ["đau buồn", "u sầu", "chán nản", "thất vọng"],
    "đau": ["đau đớn", "khổ sở", "xót xa"],
    "nhớ": ["thương nhớ", "nhung nhớ", "da diết"],
    "chán": ["buồn chán", "tẻ nhạt", "nhàm"],
    
    # Anger synonyms
    "giận": ["tức giận", "nổi giận", "bực tức"],
    "tức": ["bực mình", "khó chịu", "giận dữ"],
    "ghét": ["căm ghét", "ghê tởm", "chán ghét"],
    
    # Fear synonyms
    "sợ": ["lo sợ", "kinh sợ", "hoảng sợ"],
    "lo": ["lo lắng", "lo âu", "bồn chồn"],
    "căng thẳng": ["áp lực", "stress", "hồi hộp"],
    
    # General
    "rất": ["cực kỳ", "vô cùng", "quá", "siêu"],
    "quá": ["lắm", "ghê", "cực", "thật"],
}

# Emotion keywords for weak labeling
EMOTION_KEYWORDS = {
    "joy": {
        "strong": ["tuyệt vời", "hạnh phúc", "sung sướng", "yêu", "thích quá", 
                   "chúc mừng", "tốt quá", "hay quá", "xuất sắc"],
        "medium": ["vui", "tốt", "hay", "được", "ok", "ổn", "thích"],
        "weak": ["cũng được", "tạm", "bình thường"]
    },
    "sadness": {
        "strong": ["đau khổ", "khóc", "thất vọng quá", "buồn quá", "chán quá"],
        "medium": ["buồn", "nhớ", "tiếc", "đau", "thương"],
        "weak": ["hơi buồn", "chán", "mệt"]
    },
    "anger": {
        "strong": ["điên tiết", "giận dữ", "tức chết", "ghét cay"],
        "medium": ["tức", "giận", "bực", "khó chịu", "ghét"],
        "weak": ["hơi tức", "khó chịu", "bực mình"]
    },
    "fear": {
        "strong": ["kinh hoàng", "hoảng loạn", "sợ chết", "run rẩy"],
        "medium": ["sợ", "lo lắng", "căng thẳng", "hồi hộp"],
        "weak": ["hơi lo", "hơi sợ", "e ngại"]
    },
    "surprise": {
        "strong": ["sốc", "không tin nổi", "trời ơi"],
        "medium": ["ngạc nhiên", "bất ngờ", "wow", "ủa"],
        "weak": ["hơi ngạc nhiên", "lạ"]
    },
    "disgust": {
        "strong": ["ghê tởm", "kinh tởm", "buồn nôn"],
        "medium": ["ghê", "kinh", "dơ", "bẩn"],
        "weak": ["hơi ghê", "kỳ"]
    },
    "trust": {
        "strong": ["tin tưởng tuyệt đối", "chắc chắn"],
        "medium": ["tin", "ủng hộ", "yên tâm"],
        "weak": ["có lẽ", "được"]
    },
    "anticipation": {
        "strong": ["háo hức quá", "không chờ được"],
        "medium": ["mong", "chờ đợi", "hy vọng"],
        "weak": ["hơi mong", "đợi"]
    }
}


@dataclass
class AugmentedSample:
    """Augmented data sample with metadata."""
    original_text: str
    augmented_text: str
    augmentation_type: str
    primary_emotion: str
    intensity: int
    emoji_1: str
    emoji_2: Optional[str] = None
    emoji_3: Optional[str] = None
    confidence: float = 1.0  # 1.0 for manual, < 1.0 for weak-labeled


def synonym_replacement(text: str, n_replacements: int = 1) -> str:
    """
    Replace n words with synonyms.
    
    Args:
        text: Input text
        n_replacements: Number of words to replace
        
    Returns:
        Augmented text
    """
    words = text.split()
    new_words = words.copy()
    
    # Find replaceable words
    replaceable_indices = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in VIETNAMESE_SYNONYMS:
            replaceable_indices.append(i)
    
    if not replaceable_indices:
        return text
    
    # Randomly select words to replace
    n_to_replace = min(n_replacements, len(replaceable_indices))
    indices_to_replace = random.sample(replaceable_indices, n_to_replace)
    
    for idx in indices_to_replace:
        word = words[idx].lower()
        synonyms = VIETNAMESE_SYNONYMS.get(word, [])
        if synonyms:
            new_words[idx] = random.choice(synonyms)
    
    return " ".join(new_words)


def intensity_variation(text: str, current_intensity: int) -> Tuple[str, int]:
    """
    Create variations by adding/removing intensity modifiers.
    
    Args:
        text: Input text
        current_intensity: Current intensity level (1-5)
        
    Returns:
        Tuple of (modified_text, new_intensity)
    """
    intensifiers = ["rất", "cực kỳ", "vô cùng", "quá", "siêu"]
    weakeners = ["hơi", "một chút", "tạm"]
    
    text_lower = text.lower()
    
    # Try to increase intensity
    if current_intensity < 5:
        for intensifier in intensifiers:
            if intensifier not in text_lower:
                # Add intensifier at beginning or before adjective
                words = text.split()
                if len(words) > 1:
                    words.insert(1, intensifier)
                    return " ".join(words), min(5, current_intensity + 1)
    
    # Try to decrease intensity
    if current_intensity > 1:
        for intensifier in intensifiers:
            if intensifier in text_lower:
                new_text = text_lower.replace(intensifier, "").strip()
                new_text = re.sub(r'\s+', ' ', new_text)
                return new_text, max(1, current_intensity - 1)
    
    return text, current_intensity


def weak_label_text(text: str) -> Tuple[Optional[str], int, float]:
    """
    Automatically label text using keyword matching (weak labeling).
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (emotion, intensity, confidence)
        Returns (None, 0, 0) if no emotion detected
    """
    text_lower = text.lower()
    
    best_emotion = None
    best_intensity = 0
    best_confidence = 0.0
    
    for emotion, intensity_keywords in EMOTION_KEYWORDS.items():
        # Check strong keywords first
        for keyword in intensity_keywords["strong"]:
            if keyword in text_lower:
                if best_confidence < 0.9:
                    best_emotion = emotion
                    best_intensity = 5
                    best_confidence = 0.9
                break
        
        # Check medium keywords
        for keyword in intensity_keywords["medium"]:
            if keyword in text_lower:
                if best_confidence < 0.7:
                    best_emotion = emotion
                    best_intensity = 3
                    best_confidence = 0.7
                break
        
        # Check weak keywords
        for keyword in intensity_keywords["weak"]:
            if keyword in text_lower:
                if best_confidence < 0.5:
                    best_emotion = emotion
                    best_intensity = 2
                    best_confidence = 0.5
                break
    
    return best_emotion, best_intensity, best_confidence


def augment_dataset(
    samples: List[Dict],
    augmentation_factor: int = 2,
    include_weak_labeled: bool = True
) -> List[Dict]:
    """
    Augment dataset with synonym replacement and intensity variations.
    
    Args:
        samples: Original samples
        augmentation_factor: How many augmented samples per original
        include_weak_labeled: Whether to include weak-labeled samples
        
    Returns:
        Augmented dataset
    """
    augmented = []
    
    for sample in samples:
        # Keep original
        sample['confidence'] = 1.0
        sample['augmentation_type'] = 'original'
        augmented.append(sample.copy())
        
        text = sample['text']
        emotion = sample['primary_emotion']
        intensity = sample['intensity']
        
        # Synonym replacement
        for i in range(augmentation_factor):
            aug_text = synonym_replacement(text, n_replacements=1)
            if aug_text != text:
                aug_sample = sample.copy()
                aug_sample['text'] = aug_text
                aug_sample['confidence'] = 0.95
                aug_sample['augmentation_type'] = 'synonym'
                augmented.append(aug_sample)
        
        # Intensity variation (only if intensity can change)
        if 2 <= intensity <= 4:
            var_text, var_intensity = intensity_variation(text, intensity)
            if var_text != text:
                var_sample = sample.copy()
                var_sample['text'] = var_text
                var_sample['intensity'] = var_intensity
                var_sample['confidence'] = 0.9
                var_sample['augmentation_type'] = 'intensity'
                augmented.append(var_sample)
    
    return augmented


def generate_weak_labeled_samples(
    seed_texts: List[str],
    emoji_map: Dict[str, List[str]]
) -> List[Dict]:
    """
    Generate weak-labeled samples from unlabeled texts.
    
    Args:
        seed_texts: List of unlabeled Vietnamese texts
        emoji_map: Emotion to emoji mapping
        
    Returns:
        List of weak-labeled samples
    """
    samples = []
    
    for text in seed_texts:
        emotion, intensity, confidence = weak_label_text(text)
        
        if emotion and confidence >= 0.5:
            emojis = emoji_map.get(emotion, ["🤔", "😊", "👍"])
            
            sample = {
                'text': text,
                'primary_emotion': emotion,
                'intensity': intensity,
                'emoji_1': emojis[0] if len(emojis) > 0 else "🤔",
                'emoji_2': emojis[1] if len(emojis) > 1 else None,
                'emoji_3': emojis[2] if len(emojis) > 2 else None,
                'confidence': confidence,
                'augmentation_type': 'weak_labeled',
                'source': 'auto'
            }
            samples.append(sample)
    
    return samples


def validate_dataset(samples: List[Dict], min_confidence: float = 0.5) -> Dict:
    """
    Validate dataset quality.
    
    Args:
        samples: Dataset samples
        min_confidence: Minimum confidence threshold
        
    Returns:
        Validation report
    """
    total = len(samples)
    high_quality = sum(1 for s in samples if s.get('confidence', 1.0) >= 0.9)
    medium_quality = sum(1 for s in samples if 0.7 <= s.get('confidence', 1.0) < 0.9)
    low_quality = sum(1 for s in samples if s.get('confidence', 1.0) < 0.7)
    
    # Emotion distribution
    emotion_counts = {}
    for s in samples:
        emotion = s.get('primary_emotion', 'unknown')
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Check balance
    avg_per_emotion = total / len(emotion_counts) if emotion_counts else 0
    imbalance_ratio = max(emotion_counts.values()) / min(emotion_counts.values()) if emotion_counts else 0
    
    return {
        'total_samples': total,
        'high_quality': high_quality,
        'medium_quality': medium_quality,
        'low_quality': low_quality,
        'emotion_distribution': emotion_counts,
        'avg_per_emotion': avg_per_emotion,
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': imbalance_ratio < 2.0,
        'quality_score': (high_quality * 1.0 + medium_quality * 0.7 + low_quality * 0.4) / total if total > 0 else 0
    }


# Sample unlabeled texts for weak labeling
SAMPLE_UNLABELED_TEXTS = [
    # Joy/Happy
    "Hôm nay trời đẹp quá",
    "Được tăng lương rồi",
    "Cuối tuần đi chơi",
    "Team mình thắng rồi",
    "Ăn ngon quá",
    "Phim hay ghê",
    "Được nghỉ phép vui quá",
    "Gặp lại bạn cũ",
    "Sinh nhật vui vẻ",
    "Đạt điểm cao",
    "Trúng giải rồi",
    "Pass phỏng vấn",
    "Được thăng chức",
    "Có người yêu rồi",
    "Mua được nhà",
    "Du lịch thôi",
    "Gặp thần tượng",
    "Ngủ ngon lành",
    "Về quê ăn tết",
    "Nghỉ học hôm nay",
    "Đồ ăn ngon tuyệt",
    "Trời ơi quá may",
    "Bạn mua cho quà",
    "Thắng cược rồi",
    "Hết dịch rồi",
    "Được đi du học",
    "Pass kỳ thi",
    "Siêu thích luôn",
    
    # Sadness
    "Công việc áp lực quá",
    "Bạn bè xa dần",
    "Thức khuya quá mệt",
    "Bị cancel kế hoạch",
    "Học không hiểu gì",
    "Dọn phòng mệt ghê",
    "Thất tình rồi",
    "Bị điểm kém",
    "Mất đồ rồi",
    "Không có tiền",
    "Xa nhà quá",
    "Thời tiết xấu",
    "Hết pin rồi",
    "Không ngủ được",
    "Quên mang ví",
    "Trễ xe bus",
    "Ốm rồi",
    "Cảm thấy cô đơn",
    "Nhớ người yêu cũ",
    "Thất nghiệp rồi",
    "Phải ở nhà",
    "Lỡ chuyến bay",
    "Thất bại rồi",
    "Không ai hiểu",
    
    # Anger
    "Bị sếp mắng",
    "Deadline gấp quá",
    "Đường tắc kinh khủng",
    "Bị lừa rồi",
    "Dịch vụ tệ quá",
    "Mạng chậm ghê",
    "Bị cướp mất",
    "Bị nói xấu",
    "Không công bằng",
    "Tức mình quá",
    "Đợi lâu ghê",
    "Bị phạt tiền",
    "Làm sai quy trình",
    "Bị từ chối",
    "Chậm trễ hoài",
    "Bị hủy đơn",
    "Dịch vụ kém",
    "Không trả lời",
    
    # Fear/Anxiety
    "Mai thi rồi lo quá",
    "Chờ kết quả hồi hộp",
    "Sắp hết hạn",
    "Không kịp rồi",
    "Quên làm bài tập",
    "Bố mẹ biết mất",
    "Sắp hết tiền",
    "Không chuẩn bị",
    "Quá hạn rồi",
    "Phải họp gấp",
    "Gặp ông chủ",
    "Kiểm tra đột xuất",
    "Nguy hiểm quá",
    "Không biết làm sao",
    "Lo lắng mãi",
    "Bố mẹ giận rồi",
    
    # Surprise
    "Tin được không",
    "Không ngờ vậy",
    "Thật à",
    "Sao lại thế",
    "Trời ơi đất hỡi",
    "Bất ngờ ghê",
    "Giá rẻ vậy",
    "Nhanh vậy sao",
    "Ai ngờ được",
    "Khác hẳn",
    "Lạ thật",
    "Không tin nổi",
    "Thật sự à",
    "Ủa vậy sao",
    
    # Disgust
    "Dơ bẩn quá",
    "Khó chịu ghê",
    "Hết chỗ rồi",
    "Hỏng rồi",
    "Chất lượng tệ",
    "Không sạch sẽ",
    "Mùi khó chịu",
    "Trông kinh khủng",
    "Không thích tí nào",
    "Tệ hại thật",
    "Chán ngắt",
    "Không ưng",
    
    # Trust/Support
    "Tin bạn mà",
    "Cố gắng lên",
    "Bạn làm được",
    "Ổn mà",
    "Tin tưởng nhé",
    "Đừng lo",
    "Không sao đâu",
    "Hỗ trợ nhé",
    "Cùng nhau làm",
    "Giúp đỡ nha",
    
    # Anticipation
    "Sắp được nghỉ",
    "Chờ đợi mãi",
    "Mong tới thứ 6",
    "Sắp có kỳ nghỉ",
    "Chờ mãi",
    "Sắp đến rồi",
    "Còn vài ngày",
    "Mong đợi quá",
    "Háo hức ghê",
    "Chờ không nổi",
]


if __name__ == "__main__":
    from data.collect_data import create_initial_dataset, save_dataset_csv
    from src.models import EMOTION_EMOJI_MAP
    
    print("=" * 60)
    print("DATA AUGMENTATION & WEAK LABELING")
    print("=" * 60)
    
    # 1. Load original dataset
    original_samples = create_initial_dataset()
    print(f"\n1. Original samples: {len(original_samples)}")
    
    # 2. Augment with synonyms
    augmented_samples = augment_dataset(original_samples, augmentation_factor=2)
    print(f"2. After augmentation: {len(augmented_samples)}")
    
    # 3. Generate weak-labeled samples
    weak_labeled = generate_weak_labeled_samples(SAMPLE_UNLABELED_TEXTS, EMOTION_EMOJI_MAP)
    print(f"3. Weak-labeled samples: {len(weak_labeled)}")
    
    # 4. Combine
    all_samples = augmented_samples + weak_labeled
    print(f"4. Total samples: {len(all_samples)}")
    
    # 5. Validate
    validation = validate_dataset(all_samples)
    print(f"\n5. Validation Report:")
    print(f"   - High quality: {validation['high_quality']}")
    print(f"   - Medium quality: {validation['medium_quality']}")
    print(f"   - Low quality: {validation['low_quality']}")
    print(f"   - Quality score: {validation['quality_score']:.2%}")
    print(f"   - Balanced: {validation['is_balanced']}")
    
    # 6. Save
    save_dataset_csv(all_samples, "data/processed/augmented_data.csv")
    print(f"\n✓ Saved to data/processed/augmented_data.csv")
