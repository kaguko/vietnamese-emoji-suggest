"""
Data collection utilities for Vietnamese Emoji Suggestion System.

This module provides tools for collecting, organizing, and validating
training data for the emoji suggestion model.
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


# Emotion labels based on Plutchik's wheel of emotions
EMOTION_LABELS = {
    0: "joy",           # vui
    1: "sadness",       # buồn
    2: "anger",         # giận
    3: "fear",          # sợ
    4: "surprise",      # ngạc nhiên
    5: "disgust",       # ghê tởm
    6: "trust",         # tin tưởng
    7: "anticipation"   # mong đợi
}

EMOTION_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}

# Intensity levels
INTENSITY_LEVELS = {
    1: "very_weak",     # gần như neutral
    2: "weak",          # yếu
    3: "medium",        # trung bình
    4: "strong",        # mạnh
    5: "very_strong"    # rất mạnh
}


@dataclass
class DataSample:
    """A single data sample for training."""
    text: str
    primary_emotion: str
    intensity: int
    emoji_1: str
    emoji_2: Optional[str] = None
    emoji_3: Optional[str] = None
    source: str = "manual"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        
        # Validate emotion
        if self.primary_emotion not in EMOTION_TO_IDX:
            raise ValueError(f"Invalid emotion: {self.primary_emotion}. "
                           f"Must be one of {list(EMOTION_TO_IDX.keys())}")
        
        # Validate intensity
        if self.intensity not in INTENSITY_LEVELS:
            raise ValueError(f"Invalid intensity: {self.intensity}. Must be 1-5")


def create_initial_dataset() -> List[Dict]:
    """
    Create initial manually-curated dataset.
    This provides 100+ samples covering all 8 emotions.
    """
    samples = [
        # JOY (vui)
        DataSample("Chúc mừng bạn đậu tuyển dụng!", "joy", 5, "😊", "🎉", "🥳"),
        DataSample("Hôm nay là ngày tuyệt vời!", "joy", 4, "😄", "✨", "🌟"),
        DataSample("Cảm ơn bạn nhiều lắm!", "joy", 4, "😊", "🙏", "❤️"),
        DataSample("Mình vui quá!", "joy", 5, "😁", "🎊", "💫"),
        DataSample("Thật tuyệt vời!", "joy", 4, "🤩", "👏", "✨"),
        DataSample("Cuối cùng cũng xong!", "joy", 4, "😌", "🎉", "💪"),
        DataSample("Được nghỉ phép rồi!", "joy", 4, "😊", "🏖️", "🎊"),
        DataSample("Lương tháng này tăng!", "joy", 5, "🤑", "💰", "🎉"),
        DataSample("Con đỗ đại học rồi mẹ ơi!", "joy", 5, "😭", "🎓", "🎉"),
        DataSample("Ăn mừng thôi!", "joy", 4, "🎉", "🍻", "🥳"),
        DataSample("Vui ghê!", "joy", 3, "😊", "😄", "🙂"),
        DataSample("Có tin vui nè!", "joy", 4, "😊", "✨", "🌈"),
        
        # SADNESS (buồn)
        DataSample("Buồn quá", "sadness", 5, "😭", "😢", "💔"),
        DataSample("Mình rất nhớ bạn", "sadness", 4, "😢", "💔", "🥺"),
        DataSample("Thật đáng tiếc", "sadness", 3, "😔", "😞", "💔"),
        DataSample("Hôm nay không vui", "sadness", 3, "😔", "☹️", "😞"),
        DataSample("Thi trượt rồi", "sadness", 4, "😭", "😢", "💔"),
        DataSample("Chia tay rồi", "sadness", 5, "💔", "😢", "😭"),
        DataSample("Mất việc rồi", "sadness", 5, "😭", "😞", "💔"),
        DataSample("Cô đơn quá", "sadness", 4, "😢", "🥺", "😔"),
        DataSample("Nhớ nhà", "sadness", 4, "🥺", "😢", "🏠"),
        DataSample("Thất vọng quá", "sadness", 4, "😞", "😔", "💔"),
        DataSample("Chán ghê", "sadness", 3, "😕", "😔", "😒"),
        DataSample("Mệt mỏi lắm", "sadness", 3, "😩", "😞", "😔"),
        
        # ANGER (giận)
        DataSample("Tức quá!", "anger", 5, "😠", "💢", "😤"),
        DataSample("Sao lại thế được!", "anger", 4, "😡", "💢", "😤"),
        DataSample("Bực mình ghê!", "anger", 4, "😤", "💢", "😠"),
        DataSample("Không chấp nhận được!", "anger", 5, "😡", "👊", "💢"),
        DataSample("Quá đáng!", "anger", 4, "😠", "💢", "🤬"),
        DataSample("Ghét cay ghét đắng!", "anger", 5, "🤬", "💢", "😡"),
        DataSample("Điên tiết lên được!", "anger", 5, "🤬", "💢", "😤"),
        DataSample("Mày làm gì vậy!", "anger", 4, "😠", "💢", "😤"),
        DataSample("Khó chịu quá!", "anger", 3, "😒", "😤", "💢"),
        DataSample("Sốt ruột quá!", "anger", 3, "😤", "⏰", "😠"),
        
        # FEAR (sợ)
        DataSample("Sợ quá!", "fear", 5, "😨", "😱", "😰"),
        DataSample("Lo lắng quá!", "fear", 4, "😰", "😟", "🥺"),
        DataSample("Căng thẳng quá!", "fear", 4, "😰", "😬", "💦"),
        DataSample("Không dám đâu!", "fear", 3, "😨", "🙈", "😰"),
        DataSample("Run hết cả người!", "fear", 5, "😱", "😨", "💀"),
        DataSample("Hồi hộp quá!", "fear", 3, "😬", "💓", "😰"),
        DataSample("Mai thi rồi!", "fear", 4, "😰", "📚", "😱"),
        DataSample("Đáng sợ thật!", "fear", 4, "😱", "😨", "👻"),
        DataSample("Hoang mang quá!", "fear", 4, "😰", "😟", "❓"),
        DataSample("Không biết sao!", "fear", 3, "😰", "🤔", "😟"),
        
        # SURPRISE (ngạc nhiên)
        DataSample("Thật sao!", "surprise", 4, "😮", "😲", "🤯"),
        DataSample("Không tin được!", "surprise", 5, "😱", "🤯", "😮"),
        DataSample("Ơ kìa!", "surprise", 3, "😮", "❓", "😯"),
        DataSample("Bất ngờ quá!", "surprise", 5, "🤯", "😲", "🎉"),
        DataSample("Wow!", "surprise", 4, "🤩", "😮", "✨"),
        DataSample("Trời ơi!", "surprise", 4, "😱", "😮", "🙀"),
        DataSample("Không ngờ!", "surprise", 4, "😲", "😮", "🤯"),
        DataSample("Ủa!", "surprise", 3, "🤔", "😮", "❓"),
        DataSample("Gì đây!", "surprise", 3, "😮", "🤔", "❓"),
        DataSample("Đùa à!", "surprise", 4, "😲", "🤣", "😮"),
        
        # DISGUST (ghê tởm)
        DataSample("Ghê quá!", "disgust", 5, "🤢", "🤮", "😖"),
        DataSample("Kinh dị!", "disgust", 4, "😱", "🤢", "👎"),
        DataSample("Không chịu được!", "disgust", 4, "🤮", "😖", "❌"),
        DataSample("Dơ bẩn quá!", "disgust", 4, "🤢", "😷", "🚫"),
        DataSample("Kỳ quá!", "disgust", 3, "😒", "🙄", "😕"),
        DataSample("Ớn lạnh!", "disgust", 4, "😖", "🤢", "😬"),
        DataSample("Không thích!", "disgust", 3, "👎", "😕", "❌"),
        DataSample("Tệ quá!", "disgust", 4, "👎", "😤", "💔"),
        DataSample("Dở ẹc!", "disgust", 3, "👎", "😒", "🙄"),
        DataSample("Chán ngấy!", "disgust", 4, "😒", "🙄", "😤"),
        
        # TRUST (tin tưởng)
        DataSample("Tin bạn!", "trust", 4, "🤝", "💪", "👍"),
        DataSample("Cậu làm được!", "trust", 4, "💪", "✨", "👏"),
        DataSample("Mình ủng hộ!", "trust", 4, "👍", "💪", "🤝"),
        DataSample("Yên tâm đi!", "trust", 4, "😌", "🤗", "👌"),
        DataSample("Đáng tin cậy!", "trust", 4, "🤝", "✅", "💯"),
        DataSample("Cùng nhau nhé!", "trust", 4, "🤝", "💪", "❤️"),
        DataSample("Không lo!", "trust", 3, "👌", "😊", "✌️"),
        DataSample("Chắc chắn!", "trust", 5, "✅", "💯", "👍"),
        DataSample("Bạn giỏi lắm!", "trust", 4, "👏", "🌟", "💪"),
        DataSample("Tuyệt vời!", "trust", 4, "👍", "✨", "🔥"),
        
        # ANTICIPATION (mong đợi)
        DataSample("Mong chờ quá!", "anticipation", 4, "🤞", "😊", "✨"),
        DataSample("Háo hức quá!", "anticipation", 5, "🤩", "😆", "🎊"),
        DataSample("Không đợi được nữa!", "anticipation", 5, "😆", "🔥", "⏰"),
        DataSample("Sắp đến rồi!", "anticipation", 4, "🎉", "⏰", "✨"),
        DataSample("Chờ đợi!", "anticipation", 3, "⏰", "🤞", "😊"),
        DataSample("Hy vọng!", "anticipation", 4, "🤞", "✨", "🙏"),
        DataSample("Mai là sinh nhật!", "anticipation", 5, "🎂", "🎉", "🤩"),
        DataSample("Còn 3 ngày nữa!", "anticipation", 4, "⏳", "🤞", "😊"),
        DataSample("Cuối tuần rồi!", "anticipation", 4, "🎉", "🥳", "✨"),
        DataSample("Sắp nghỉ hè!", "anticipation", 5, "🏖️", "☀️", "🎉"),
        
        # Additional JOY samples
        DataSample("Được thưởng rồi!", "joy", 5, "🤑", "🎉", "💰"),
        DataSample("Pass rồi!", "joy", 5, "🎉", "😊", "✨"),
        DataSample("Được tăng ca!", "joy", 3, "😊", "💼", "👍"),
        DataSample("Trúng số rồi!", "joy", 5, "🤩", "💰", "🎊"),
        DataSample("Được khen!", "joy", 4, "😊", "🌟", "👏"),
        
        # Additional SADNESS samples
        DataSample("Ế quá!", "sadness", 4, "😔", "💔", "😢"),
        DataSample("Không ai hiểu", "sadness", 4, "😢", "💔", "😞"),
        DataSample("Hụt hẫng", "sadness", 3, "😔", "💔", "😕"),
        DataSample("Trượt visa rồi", "sadness", 5, "😭", "😞", "💔"),
        DataSample("Bị từ chối", "sadness", 4, "😢", "💔", "😔"),
        
        # Additional ANGER samples
        DataSample("Phí thời gian!", "anger", 4, "😤", "💢", "⏰"),
        DataSample("Vô trách nhiệm!", "anger", 5, "😡", "💢", "👊"),
        DataSample("Lừa dối!", "anger", 5, "😠", "💢", "😤"),
        DataSample("Bất công!", "anger", 4, "😡", "💢", "😤"),
        DataSample("Thái độ tệ!", "anger", 4, "😠", "💢", "👎"),
        
        # Additional FEAR samples
        DataSample("Nguy hiểm quá!", "fear", 5, "😱", "😨", "⚠️"),
        DataSample("Không chắc!", "fear", 3, "😰", "🤔", "😟"),
        DataSample("Lo âu quá!", "fear", 4, "😰", "😟", "💭"),
        DataSample("Sợ bị phạt!", "fear", 4, "😨", "😰", "⚠️"),
        DataSample("Cẩn thận!", "fear", 3, "😬", "⚠️", "👀"),
        
        # Additional SURPRISE samples
        DataSample("Không thể tin!", "surprise", 5, "😱", "🤯", "❗"),
        DataSample("Thật ư!", "surprise", 4, "😮", "😲", "❓"),
        DataSample("Sao lại vậy!", "surprise", 4, "😮", "😲", "❗"),
        DataSample("À!", "surprise", 3, "😮", "💡", "🤔"),
        DataSample("Bất ngờ thật!", "surprise", 4, "😲", "😮", "✨"),
        
        # Additional DISGUST samples
        DataSample("Hôi quá!", "disgust", 4, "🤢", "😷", "👃"),
        DataSample("Kinh tởm!", "disgust", 5, "🤮", "😖", "🤢"),
        DataSample("Ghê tởm quá!", "disgust", 5, "🤢", "🤮", "😷"),
        DataSample("Không ổn!", "disgust", 3, "😕", "👎", "❌"),
        DataSample("Rất tệ!", "disgust", 4, "👎", "😤", "🚫"),
        
        # Additional TRUST samples
        DataSample("Cố lên!", "trust", 4, "💪", "✨", "👊"),
        DataSample("Được đó!", "trust", 3, "👍", "😊", "✅"),
        DataSample("OK nhé!", "trust", 3, "👌", "😊", "✌️"),
        DataSample("Hoàn hảo!", "trust", 5, "💯", "✨", "🌟"),
        DataSample("Tốt lắm!", "trust", 4, "👍", "👏", "😊"),
        
        # Additional ANTICIPATION samples
        DataSample("Sắp tết rồi!", "anticipation", 5, "🎉", "🧧", "🎊"),
        DataSample("Chờ đợi kết quả!", "anticipation", 4, "⏰", "🤞", "📋"),
        DataSample("Sắp gặp!", "anticipation", 4, "😊", "⏰", "🤗"),
        DataSample("Mong được đi!", "anticipation", 4, "🤞", "✈️", "😊"),
        DataSample("Sắp có tin!", "anticipation", 4, "⏰", "🤞", "📱"),
        
        # More JOY samples
        DataSample("Ngon quá!", "joy", 4, "😋", "👍", "✨"),
        DataSample("Tốt đẹp!", "joy", 4, "😊", "✨", "👌"),
        DataSample("Được rồi!", "joy", 3, "😊", "👍", "✅"),
        DataSample("Vui lắm!", "joy", 4, "😄", "😊", "🎉"),
        DataSample("Xinh quá!", "joy", 4, "😍", "✨", "💕"),
        DataSample("Đẹp tuyệt!", "joy", 5, "🤩", "✨", "😍"),
        DataSample("Cảm động quá!", "joy", 5, "😭", "❤️", "🥺"),
        DataSample("Hài lòng!", "joy", 4, "😊", "👍", "✨"),
        
        # More SADNESS samples
        DataSample("Buồn thiu!", "sadness", 4, "😔", "😢", "💔"),
        DataSample("Đau lòng!", "sadness", 5, "💔", "😭", "😢"),
        DataSample("Xót xa!", "sadness", 5, "😢", "💔", "😭"),
        DataSample("Tiếc nuối!", "sadness", 4, "😔", "😢", "💔"),
        DataSample("Thương quá!", "sadness", 4, "🥺", "😢", "💔"),
        DataSample("Mệt nản!", "sadness", 4, "😩", "😔", "😞"),
        DataSample("Chán nản!", "sadness", 3, "😔", "😕", "😒"),
        DataSample("Tuyệt vọng!", "sadness", 5, "😭", "😞", "💔"),
        
        # More ANGER samples
        DataSample("Tức lắm!", "anger", 4, "😤", "💢", "😠"),
        DataSample("Ức chế!", "anger", 4, "😤", "💢", "😡"),
        DataSample("Chướng tai!", "anger", 3, "😒", "😤", "💢"),
        DataSample("Không ưa!", "anger", 3, "😒", "👎", "😤"),
        DataSample("Phẫn nộ!", "anger", 5, "😡", "💢", "🤬"),
        DataSample("Bực bội!", "anger", 4, "😤", "😠", "💢"),
        DataSample("Nóng giận!", "anger", 5, "😡", "💢", "🔥"),
        DataSample("Chán không!", "anger", 3, "😒", "😤", "😕"),
        
        # More FEAR samples
        DataSample("Khủng khiếp!", "fear", 5, "😱", "😨", "😰"),
        DataSample("Rùng mình!", "fear", 4, "😨", "😰", "🥶"),
        DataSample("E dè!", "fear", 3, "😰", "😟", "🤔"),
        DataSample("Bất an!", "fear", 4, "😰", "😟", "💭"),
        DataSample("Lo lắng!", "fear", 4, "😰", "😟", "😬"),
        DataSample("Hoảng hốt!", "fear", 5, "😱", "😨", "😰"),
        DataSample("Run sợ!", "fear", 5, "😱", "😨", "🥶"),
        DataSample("Ngại ngùng!", "fear", 3, "😰", "😊", "🙈"),
        
        # More SURPRISE samples
        DataSample("Sốc nặng!", "surprise", 5, "😱", "🤯", "😲"),
        DataSample("Ngỡ ngàng!", "surprise", 4, "😮", "😲", "🤯"),
        DataSample("Kinh ngạc!", "surprise", 5, "😲", "🤯", "😮"),
        DataSample("Lạ thật!", "surprise", 3, "😮", "🤔", "😯"),
        DataSample("Ồ!", "surprise", 3, "😮", "😲", "❗"),
        DataSample("Hả!", "surprise", 3, "😮", "😯", "❓"),
        DataSample("Lạ lùng!", "surprise", 4, "😮", "🤔", "😲"),
        DataSample("Chưa thấy!", "surprise", 4, "😮", "😲", "🤯"),
        
        # More DISGUST samples
        DataSample("Phản cảm!", "disgust", 4, "😖", "😒", "🙄"),
        DataSample("Thất vọng!", "disgust", 4, "😞", "👎", "😔"),
        DataSample("Chất lượng kém!", "disgust", 4, "👎", "😤", "😒"),
        DataSample("Tồi tệ!", "disgust", 5, "😖", "🤮", "👎"),
        DataSample("Kém cỏi!", "disgust", 4, "😒", "👎", "😤"),
        DataSample("Nhạt nhẽo!", "disgust", 3, "😒", "😕", "👎"),
        DataSample("Khó ưa!", "disgust", 4, "😒", "😖", "👎"),
        DataSample("Chịu không nổi!", "disgust", 5, "🤢", "🤮", "😖"),
        
        # More TRUST samples
        DataSample("Tin được!", "trust", 4, "👍", "✅", "😊"),
        DataSample("Yên tâm!", "trust", 4, "😌", "👌", "💪"),
        DataSample("Đồng ý!", "trust", 4, "👍", "✅", "🤝"),
        DataSample("Hợp lý!", "trust", 3, "👍", "🤔", "✅"),
        DataSample("Đúng rồi!", "trust", 4, "✅", "👍", "💯"),
        DataSample("Làm được!", "trust", 4, "💪", "👍", "✨"),
        DataSample("Chắc ăn!", "trust", 5, "💯", "✅", "👍"),
        DataSample("Ổn áp!", "trust", 3, "👌", "😊", "👍"),
        
        # More ANTICIPATION samples  
        DataSample("Sắp rồi!", "anticipation", 4, "⏰", "😊", "✨"),
        DataSample("Đợi mãi!", "anticipation", 4, "⏳", "😊", "🤞"),
        DataSample("Mong ngóng!", "anticipation", 4, "🤞", "⏰", "😊"),
        DataSample("Hồi hộp chờ!", "anticipation", 4, "😬", "🤞", "⏰"),
        DataSample("Sắp được nghỉ!", "anticipation", 5, "🎉", "🏖️", "😊"),
        DataSample("Gần đến!", "anticipation", 4, "⏰", "✨", "😊"),
        DataSample("Mong mỏi!", "anticipation", 4, "🤞", "✨", "😊"),
        DataSample("Háo hức!", "anticipation", 5, "🤩", "🎉", "✨"),
        
        # Extra samples to reach target
        DataSample("Tuyệt vời quá!", "joy", 5, "🤩", "🎉", "✨"),
        DataSample("Quá vui!", "joy", 5, "😄", "🎉", "😊"),
        DataSample("Rất buồn!", "sadness", 5, "😭", "😢", "💔"),
        DataSample("Cực kỳ tức giận!", "anger", 5, "😡", "💢", "🤬"),
        DataSample("Sợ hãi!", "fear", 5, "😱", "😨", "😰"),
        DataSample("Thật ngạc nhiên!", "surprise", 5, "😮", "😲", "🤯"),
    ]
    
    return [asdict(s) for s in samples]


def save_dataset_csv(samples: List[Dict], filepath: str):
    """Save dataset to CSV file."""
    if not samples:
        print("No samples to save!")
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fieldnames = list(samples[0].keys())
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"Saved {len(samples)} samples to {filepath}")


def save_dataset_json(samples: List[Dict], filepath: str):
    """Save dataset to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(samples)} samples to {filepath}")


def load_dataset_csv(filepath: str) -> List[Dict]:
    """Load dataset from CSV file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['intensity'] = int(row['intensity'])
            samples.append(row)
    return samples


def load_dataset_json(filepath: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dataset_stats(samples: List[Dict]) -> Dict:
    """Get statistics about the dataset."""
    stats = {
        'total_samples': len(samples),
        'emotions': {},
        'intensities': {},
        'avg_text_length': 0,
        'emoji_counts': {}
    }
    
    total_length = 0
    for sample in samples:
        # Count emotions
        emotion = sample['primary_emotion']
        stats['emotions'][emotion] = stats['emotions'].get(emotion, 0) + 1
        
        # Count intensities
        intensity = sample['intensity']
        stats['intensities'][intensity] = stats['intensities'].get(intensity, 0) + 1
        
        # Text length
        total_length += len(sample['text'].split())
        
        # Count emojis
        for key in ['emoji_1', 'emoji_2', 'emoji_3']:
            emoji = sample.get(key)
            if emoji:
                stats['emoji_counts'][emoji] = stats['emoji_counts'].get(emoji, 0) + 1
    
    stats['avg_text_length'] = total_length / len(samples) if samples else 0
    
    return stats


def validate_dataset(samples: List[Dict]) -> Dict:
    """Validate dataset for common issues."""
    issues = {
        'missing_emoji': [],
        'invalid_emotion': [],
        'invalid_intensity': [],
        'duplicate_text': [],
        'short_text': []
    }
    
    seen_texts = set()
    
    for i, sample in enumerate(samples):
        # Check for missing primary emoji
        if not sample.get('emoji_1'):
            issues['missing_emoji'].append(i)
        
        # Check emotion validity
        if sample.get('primary_emotion') not in EMOTION_TO_IDX:
            issues['invalid_emotion'].append(i)
        
        # Check intensity validity
        if sample.get('intensity') not in INTENSITY_LEVELS:
            issues['invalid_intensity'].append(i)
        
        # Check for duplicates
        text = sample.get('text', '').strip().lower()
        if text in seen_texts:
            issues['duplicate_text'].append(i)
        seen_texts.add(text)
        
        # Check for very short texts
        if len(text.split()) < 2:
            issues['short_text'].append(i)
    
    return issues


if __name__ == "__main__":
    # Create initial dataset
    print("Creating initial dataset...")
    samples = create_initial_dataset()
    
    # Get stats
    stats = get_dataset_stats(samples)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Emotions: {stats['emotions']}")
    print(f"  Avg text length: {stats['avg_text_length']:.1f} words")
    
    # Validate
    issues = validate_dataset(samples)
    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        print("\n✓ Dataset validation passed!")
    else:
        print(f"\n⚠ Found {total_issues} issues:")
        for issue_type, indices in issues.items():
            if indices:
                print(f"  {issue_type}: {len(indices)} samples")
    
    # Save to files
    save_dataset_csv(samples, "data/raw/initial_data.csv")
    save_dataset_json(samples, "data/raw/initial_data.json")
    
    print("\n✓ Initial dataset created successfully!")
