# Data Directory

This directory contains all data for the Vietnamese Emoji Suggestion System.

## Structure

```
data/
├── raw/                    # Original collected data
│   ├── initial_data.csv    # Initial 100+ samples
│   └── initial_data.json   # JSON format
├── processed/              # Cleaned and labeled data
│   └── labeled_data.csv    # Final labeled dataset
├── collect_data.py         # Data collection utilities
└── README.md               # This file
```

## Data Format

### CSV Format
```csv
text,primary_emotion,intensity,emoji_1,emoji_2,emoji_3,source,created_at
"Chúc mừng bạn!",joy,4,😊,🎉,🥳,manual,2025-01-09T10:00:00
```

### Fields
- `text`: Vietnamese text (string)
- `primary_emotion`: One of 8 emotions (joy, sadness, anger, fear, surprise, disgust, trust, anticipation)
- `intensity`: 1-5 scale (1=very weak, 5=very strong)
- `emoji_1`: Primary suggested emoji
- `emoji_2`: Secondary suggested emoji (optional)
- `emoji_3`: Tertiary suggested emoji (optional)
- `source`: Data source (manual, social_media, forum, etc.)
- `created_at`: Timestamp

## Creating Initial Data

```bash
python data/collect_data.py
```

This will create `data/raw/initial_data.csv` with 100+ samples.

## Emotion Categories

Based on Plutchik's Wheel of Emotions:

| Emotion | Vietnamese | Description |
|---------|------------|-------------|
| joy | Vui | Happiness, excitement |
| sadness | Buồn | Sorrow, grief |
| anger | Giận | Frustration, rage |
| fear | Sợ | Anxiety, worry |
| surprise | Ngạc nhiên | Shock, amazement |
| disgust | Ghê tởm | Revulsion, distaste |
| trust | Tin tưởng | Confidence, faith |
| anticipation | Mong đợi | Expectation, hope |
