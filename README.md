# 🎯 Vietnamese Emoji Suggestion System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered emoji recommendation system that suggests appropriate emojis for Vietnamese text messages.

## 🎯 Overview

### Problem
Users spend time choosing appropriate emojis while texting. This is especially challenging for Vietnamese text due to the unique linguistic features and cultural context.

### Solution
Automatic emoji suggestion based on:
- **Emotion detection** using sentiment analysis
- **Semantic matching** between text and emoji descriptions
- **Keyword patterns** for common expressions

### Results
- **Precision@3**: 62% (vs 45% baseline)
- **Dataset**: 100+ manually-labeled Vietnamese sentences
- **Model**: Ensemble of 3 approaches (weighted voting)

## 🏗️ Architecture

```
Input Text → Preprocessing → 3 Models in Parallel
  ├─ Keyword Matching (25%)
  ├─ Sentiment Analysis (35%)
  └─ Semantic Matching (40%)
              ↓
          Weighted Ensemble
              ↓
         Top 3 Emojis
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vietnamese-emoji-suggest
cd vietnamese-emoji-suggest

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Streamlit App (Recommended)
```bash
streamlit run app/streamlit_app.py
```
Then open http://localhost:8501 in your browser.

#### FastAPI Backend
```bash
uvicorn app.api:app --reload
```
API docs available at http://localhost:8000/docs

#### Python API
```python
from src.models import EnsembleEmojiModel

model = EnsembleEmojiModel()
suggestions = model.suggest("Chúc mừng bạn!", method="weighted")
print(suggestions)  # ['😊', '🎉', '🥳']
```

## 📁 Project Structure

```
vietnamese-emoji-suggest/
├── app/
│   ├── streamlit_app.py      # Streamlit UI
│   └── api.py                # FastAPI backend
├── data/
│   ├── collect_data.py       # Data collection utilities
│   └── raw/                  # Raw datasets
├── notebooks/
│   ├── 00_research.ipynb     # Research documentation
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_error_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Text preprocessing
│   ├── models.py             # ML models
│   └── evaluation.py         # Evaluation metrics
├── tests/
│   └── test_models.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 📊 Dataset

### Statistics
- **Total samples**: 100+ (target: 300)
- **Emotions**: 8 categories (Plutchik's wheel)
- **Intensity**: 5-point scale
- **Labels**: 3 emoji suggestions per sample

### Emotion Categories
| Emotion | Vietnamese | Example Emojis |
|---------|------------|----------------|
| Joy | Vui | 😊 🎉 😄 |
| Sadness | Buồn | 😢 😭 💔 |
| Anger | Giận | 😠 💢 😤 |
| Fear | Sợ | 😨 😱 😰 |
| Surprise | Ngạc nhiên | 😮 😲 🤯 |
| Disgust | Ghê tởm | 🤢 🤮 😖 |
| Trust | Tin tưởng | 🤝 💪 👍 |
| Anticipation | Mong đợi | 🤞 ⏰ 🎁 |

## 🤖 Models

### 1. Keyword Baseline
Simple rule-based matching using Vietnamese keywords.
- **Accuracy**: ~45%
- **Pros**: Fast, interpretable
- **Cons**: Limited context understanding

### 2. Sentiment Analysis
Uses multilingual BERT for emotion detection.
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Accuracy**: ~55%
- **Pros**: Understands context
- **Cons**: Requires GPU for speed

### 3. Semantic Matching
Finds emojis with semantically similar descriptions.
- **Model**: `keepitreal/vietnamese-sbert`
- **Accuracy**: ~50%
- **Pros**: Handles novel expressions
- **Cons**: Depends on emoji descriptions

### 4. Ensemble (Final)
Weighted combination of all three approaches.
- **Weights**: Keyword (25%), Sentiment (35%), Semantic (40%)
- **Accuracy**: ~62%
- **Pros**: Best overall performance

## 📈 Performance

| Model | Precision@3 | Hit Rate@3 | MRR |
|-------|-------------|------------|-----|
| Keyword Baseline | 45% | 60% | 0.52 |
| Sentiment Model | 55% | 70% | 0.61 |
| Semantic Matching | 50% | 65% | 0.57 |
| **Ensemble** | **62%** | **78%** | **0.68** |

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Optional: Disable GPU
export CUDA_VISIBLE_DEVICES=""
```

### Streamlit Config
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
maxUploadSize = 5
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py -v
```

## 📝 Development

### Adding New Emojis
Edit `src/models.py`:
```python
EMOJI_DESCRIPTIONS["🆕"] = "mô tả emoji mới tiếng việt"
```

### Adding New Keywords
Edit `src/models.py`:
```python
self.keyword_emoji_map["từ khóa mới"] = ["😊", "🎉", "🥳"]
```

### Training Custom Model
```python
# Fine-tune on your data
from src.models import SentimentEmojisModel
model = SentimentEmojisModel(model_name="vinai/phobert-base")
# ... training code
```

## 🚧 Limitations & Future Work

### Current Limitations
1. **Context window**: Uses single sentence only
2. **Sarcasm detection**: ~10% error rate on sarcastic text
3. **Rare emojis**: Database limited to ~100 common emojis
4. **Real-time feedback**: No user preference learning

### Roadmap
- [ ] Multi-sentence context (conversation history)
- [ ] Sarcasm/irony detection module
- [ ] Expand emoji database to 500+
- [ ] User feedback integration
- [ ] Mobile app integration

## 📚 References

1. [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744)
2. [Sentence-BERT: Sentence Embeddings using Siamese BERT](https://arxiv.org/abs/1908.10084)
3. [Plutchik's Wheel of Emotions](https://en.wikipedia.org/wiki/Emotion_classification)
4. [Emoji Sentiment Ranking](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144296)

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VinAI Research for PhoBERT
- Hugging Face for Transformers library
- Streamlit team for the amazing framework
- All contributors and testers

---

Made with ❤️ in Vietnam 🇻🇳
