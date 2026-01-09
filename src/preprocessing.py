"""
Text preprocessing module for Vietnamese Emoji Suggestion System.

This module handles:
- Teencode/slang normalization
- URL and special character removal
- Text cleaning and normalization
- Word segmentation (optional)
"""

import re
from typing import Optional, Dict, List
import unicodedata


# Comprehensive teencode dictionary for Vietnamese
TEENCODE_MAP = {
    # Common abbreviations
    "chằn zn": "trần trụi",
    "xỉu": "ngất",
    "sảng": "đau đầu",
    "gắt": "khắt khe",
    "chán chê": "chán ngán",
    
    # Common slang
    "ko": "không",
    "k": "không",
    "kh": "không",
    "khg": "không",
    "hok": "không",
    "hem": "không",
    "hông": "không",
    
    "dc": "được",
    "đc": "được",
    "dk": "được",
    "đk": "được",
    "duoc": "được",
    
    "bn": "bạn",
    "bạn": "bạn",
    
    "mk": "mình",
    "mik": "mình",
    "mìk": "mình",
    
    "ng": "người",
    "ngta": "người ta",
    "ns": "nói",
    
    "r": "rồi",
    "oy": "rồi",
    "rui": "rồi",
    
    "trc": "trước",
    "truoc": "trước",
    "trc khi": "trước khi",
    
    "sao đó": "sau đó",
    "sd": "sau đó",
    
    "bt": "biết",
    "bít": "biết",
    "biet": "biết",
    "bjt": "biết",
    
    "nte": "như thế",
    "ntn": "như thế nào",
    "sao": "sao",
    
    "cx": "cũng",
    "cg": "cũng",
    "cug": "cũng",
    
    "vs": "với",
    "voi": "với",
    "vk": "vợ",
    "ck": "chồng",
    
    "thik": "thích",
    "thix": "thích",
    "thick": "thích",
    
    "iu": "yêu",
    "iu qua": "yêu quá",
    "yeu": "yêu",
    
    "bh": "bao giờ",
    "bjh": "bao giờ",
    
    "lm": "làm",
    "lam": "làm",
    
    "nc": "nước",
    "nuoc": "nước",
    
    "nh": "nhiều",
    "nhiu": "nhiều",
    
    "qa": "quá",
    "qua": "quá",
    
    "thi": "thì",
    "thi": "thì",
    
    "đag": "đang",
    "dang": "đang",
    "dag": "đang",
    
    "hđ": "hoạt động",
    "hd": "hoạt động",
    
    "z": "vậy",
    "v": "vậy",
    "vay": "vậy",
    
    "nyc": "người yêu cũ",
    "ny": "người yêu",
    
    "sg": "Sài Gòn",
    "hn": "Hà Nội",
    
    "ah": "à",
    "ak": "ạ",
    "a": "anh",
    "e": "em",
    
    "t": "tao",
    "m": "mày",
    
    "vl": "vãi",
    "vcl": "vãi",
    "vll": "vãi",
    
    "cmn": "con mẹ nó",
    "dm": "đù má",
    
    "gato": "ghen ăn tức ở",
    
    "ok": "được",
    "okie": "được",
    "oke": "được",
    "okla": "được",
    
    "bye": "tạm biệt",
    "bai": "tạm biệt",
    "bye bye": "tạm biệt",
    
    "hi": "xin chào",
    "hello": "xin chào",
    "hellu": "xin chào",
    
    "thks": "cảm ơn",
    "thanks": "cảm ơn",
    "tks": "cảm ơn",
    "thank you": "cảm ơn",
    
    "sorry": "xin lỗi",
    "sr": "xin lỗi",
    "sry": "xin lỗi",
    
    "plz": "làm ơn",
    "pls": "làm ơn",
    "please": "làm ơn",
    
    "lol": "haha",
    "hehe": "haha",
    "hihi": "haha",
    "kk": "haha",
    "huhu": "buồn",
    
    "gì z": "gì vậy",
    "gi z": "gì vậy",
    "j z": "gì vậy",
    "j v": "gì vậy",
    
    "đẹp zai": "đẹp trai",
    "dep zai": "đẹp trai",
    "đẹp gái": "xinh gái",
    
    "real": "thật",
    "fake": "giả",
    
    "pro": "giỏi",
    "noob": "gà",
    
    "hot": "nóng bỏng",
    "cool": "tuyệt",
    "cute": "dễ thương",
    
    "wtf": "cái gì",
    "omg": "trời ơi",
    
    "sến": "sến",
    "ngầu": "ngầu",
    "chất": "chất",
    "xịn": "xịn",
    "max": "tối đa",
}

# Emotion-related keywords for reference
EMOTION_KEYWORDS = {
    "joy": ["vui", "hạnh phúc", "sung sướng", "tuyệt vời", "tốt", "hay", 
            "chúc mừng", "yêu", "thích", "cảm ơn", "giỏi", "xuất sắc"],
    "sadness": ["buồn", "đau", "khổ", "thất vọng", "nhớ", "cô đơn", 
                "chán", "mệt", "thương", "tiếc", "tội nghiệp"],
    "anger": ["giận", "tức", "bực", "khó chịu", "ghét", "ức", 
              "điên", "sốt ruột", "chán", "quá đáng"],
    "fear": ["sợ", "lo", "hoang mang", "căng thẳng", "run", "hồi hộp",
             "kinh", "đáng sợ", "rợn"],
    "surprise": ["ngạc nhiên", "bất ngờ", "sốc", "không ngờ", "wow", 
                 "ủa", "trời ơi", "thật sao"],
    "disgust": ["ghê", "kinh", "tởm", "dơ", "bẩn", "ghét", "chán",
                "kỳ", "dở", "tệ"],
    "trust": ["tin", "ủng hộ", "yên tâm", "chắc chắn", "đáng tin",
              "giỏi", "tốt", "được"],
    "anticipation": ["mong", "chờ", "háo hức", "hy vọng", "sắp",
                     "còn", "đợi", "nóng lòng"]
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to NFC form."""
    return unicodedata.normalize('NFC', text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    email_pattern = r'\S+@\S+\.\S+'
    return re.sub(email_pattern, '', text)


def remove_mentions(text: str) -> str:
    """Remove @mentions from text."""
    return re.sub(r'@\w+', '', text)


def remove_hashtags(text: str) -> str:
    """Remove #hashtags from text."""
    return re.sub(r'#\w+', '', text)


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace and normalize spaces."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_punctuation_except_basic(text: str) -> str:
    """Remove punctuation except basic sentence-ending marks."""
    # Keep: . , ! ? and Vietnamese diacritics
    text = re.sub(r'[^\w\s.,!?\u00C0-\u024F\u1E00-\u1EFF]', '', text)
    return text


def normalize_repeated_chars(text: str) -> str:
    """Normalize repeated characters (e.g., 'vuiiiii' -> 'vui')."""
    # Reduce 3+ repeated chars to 2
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


def replace_teencode(text: str, teencode_dict: Optional[Dict[str, str]] = None) -> str:
    """Replace teencode/slang with standard Vietnamese."""
    if teencode_dict is None:
        teencode_dict = TEENCODE_MAP
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_teencode = sorted(teencode_dict.items(), key=lambda x: len(x[0]), reverse=True)
    
    for slang, formal in sorted_teencode:
        # Case-insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, formal, text, flags=re.IGNORECASE)
    
    return text


def extract_emojis(text: str) -> List[str]:
    """Extract all emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.findall(text)


def remove_emojis(text: str) -> str:
    """Remove all emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_url: bool = True,
    remove_email: bool = True,
    remove_mention: bool = True,
    remove_hashtag: bool = True,
    normalize_teencode: bool = True,
    normalize_unicode_chars: bool = True,
    normalize_repeated: bool = True,
    remove_emoji: bool = True,
    custom_teencode_dict: Optional[Dict[str, str]] = None
) -> str:
    """
    Full preprocessing pipeline for Vietnamese text.
    
    Args:
        text: Input text to preprocess
        lowercase: Convert to lowercase
        remove_url: Remove URLs
        remove_email: Remove email addresses
        remove_mention: Remove @mentions
        remove_hashtag: Remove #hashtags
        normalize_teencode: Replace slang/teencode
        normalize_unicode_chars: Normalize Unicode to NFC
        normalize_repeated: Reduce repeated characters
        remove_emoji: Remove emoji characters
        custom_teencode_dict: Custom teencode mapping
    
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Unicode normalization first
    if normalize_unicode_chars:
        text = normalize_unicode(text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    if remove_url:
        text = remove_urls(text)
    
    # Remove emails
    if remove_email:
        text = remove_emails(text)
    
    # Remove mentions
    if remove_mention:
        text = remove_mentions(text)
    
    # Remove hashtags
    if remove_hashtag:
        text = remove_hashtags(text)
    
    # Remove emojis
    if remove_emoji:
        text = remove_emojis(text)
    
    # Normalize repeated characters
    if normalize_repeated:
        text = normalize_repeated_chars(text)
    
    # Replace teencode
    if normalize_teencode:
        text = replace_teencode(text, custom_teencode_dict)
    
    # Clean up whitespace
    text = remove_extra_whitespace(text)
    
    return text


class TextPreprocessor:
    """
    Text preprocessor class with configurable options.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_url: bool = True,
        remove_email: bool = True,
        remove_mention: bool = True,
        remove_hashtag: bool = True,
        normalize_teencode: bool = True,
        normalize_unicode_chars: bool = True,
        normalize_repeated: bool = True,
        remove_emoji: bool = True,
        custom_teencode_dict: Optional[Dict[str, str]] = None
    ):
        self.lowercase = lowercase
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.remove_mention = remove_mention
        self.remove_hashtag = remove_hashtag
        self.normalize_teencode = normalize_teencode
        self.normalize_unicode_chars = normalize_unicode_chars
        self.normalize_repeated = normalize_repeated
        self.remove_emoji = remove_emoji
        self.teencode_dict = custom_teencode_dict or TEENCODE_MAP
    
    def preprocess(self, text: str) -> str:
        """Preprocess a single text."""
        return preprocess_text(
            text,
            lowercase=self.lowercase,
            remove_url=self.remove_url,
            remove_email=self.remove_email,
            remove_mention=self.remove_mention,
            remove_hashtag=self.remove_hashtag,
            normalize_teencode=self.normalize_teencode,
            normalize_unicode_chars=self.normalize_unicode_chars,
            normalize_repeated=self.normalize_repeated,
            remove_emoji=self.remove_emoji,
            custom_teencode_dict=self.teencode_dict
        )
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Test preprocessing
    test_cases = [
        "Chúc mừng bạn! 🎉🎊",
        "Ko bít sao lun huhu 😢",
        "Check out https://example.com @friend #happy",
        "Vui qaaaaaa!!! 😊😊😊",
        "Thik đc iu qaaa ❤️❤️❤️",
        "Buồn quá đi mất thui 😭",
    ]
    
    print("=== PREPROCESSING TESTS ===\n")
    
    preprocessor = TextPreprocessor()
    
    for text in test_cases:
        processed = preprocessor.preprocess(text)
        emojis = extract_emojis(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print(f"Emojis extracted: {emojis}")
        print("-" * 50)
