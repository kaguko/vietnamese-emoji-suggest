# Dataset Generation Scripts

This directory contains scripts for generating and verifying the training datasets.

## Scripts

### 1. Generate Initial Dataset

Located at: `data/collect_data.py`

Generates the initial manually-curated dataset with 100+ samples.

```bash
python data/collect_data.py
```

**Outputs:**
- `data/raw/initial_data.csv` - 104 samples covering all 8 emotions
- `data/raw/initial_data.json` - Same data in JSON format

### 2. Generate Full Augmented Dataset

```bash
python scripts/generate_full_dataset.py
```

This script:
1. Loads initial data from `data/raw/initial_data.csv`
2. Applies synonym replacement augmentation
3. Applies intensity variation augmentation
4. Generates weak-labeled samples from unlabeled texts
5. Saves the full augmented dataset

**Output:**
- `data/raw/augmented_data.csv` - 450+ samples (471 generated)

**Augmentation Techniques:**
- **Synonym replacement**: Replaces Vietnamese words with synonyms
- **Intensity variations**: Modifies intensity modifiers
- **Weak labeling**: Auto-labels unlabeled texts using keyword matching

### 3. Verify Data

```bash
python scripts/verify_data.py
```

Verification checks:
- ✓ Folder structure exists
- ✓ Initial dataset has 100+ samples
- ✓ Augmented dataset has 450+ samples
- ✓ All 8 emotions are represented
- ✓ Required CSV columns present
- ✓ Data quality (no nulls, valid ranges)

## Dataset Columns

All CSV files have these columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `text` | Vietnamese text message | string | "Chúc mừng bạn!" |
| `primary_emotion` | One of 8 emotions | string | "joy" |
| `intensity` | Emotion intensity | int (1-5) | 4 |
| `emoji_1` | Primary emoji suggestion | string | "😊" |
| `emoji_2` | Secondary emoji (optional) | string | "🎉" |
| `emoji_3` | Tertiary emoji (optional) | string | "🥳" |
| `source` | Data source | string | "manual" / "augmented" |
| `created_at` | Timestamp | ISO 8601 | "2026-01-09T05:21:45" |

## Emotion Categories

The dataset covers 8 emotions based on Plutchik's wheel:

1. **joy** (vui) - 😊 🎉 😄
2. **sadness** (buồn) - 😢 😭 💔
3. **anger** (giận) - 😠 💢 😤
4. **fear** (sợ) - 😨 😱 😰
5. **surprise** (ngạc nhiên) - 😮 😲 🤯
6. **disgust** (ghê tởm) - 🤢 🤮 😖
7. **trust** (tin tưởng) - 🤝 💪 👍
8. **anticipation** (mong đợi) - 🤞 ⏰ 🎉

## Quick Start

To generate all datasets from scratch:

```bash
# 1. Generate initial dataset
python data/collect_data.py

# 2. Generate augmented dataset
python scripts/generate_full_dataset.py

# 3. Verify everything
python scripts/verify_data.py
```

Expected output:
- ✅ 104 initial samples
- ✅ 471 augmented samples
- ✅ All 8 emotions represented
- ✅ All verification checks pass
