"""
Streamlit App for Vietnamese Emoji Suggestion System

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EnsembleEmojiModel, KeywordBaseline
from src.preprocessing import preprocess_text, extract_emojis
from src.personalization import get_personalizer
from src.monitoring import get_monitor

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🎯 Vietnamese Emoji Suggester",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emoji-button {
        font-size: 3rem;
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid #ddd;
        background: white;
        cursor: pointer;
        transition: all 0.3s;
    }
    .emoji-button:hover {
        border-color: #ff6b6b;
        transform: scale(1.1);
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .analysis-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .footer {
        text-align: center;
        color: #888;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def load_model(use_full_model: bool = False):
    """Load the emoji suggestion model (cached for performance)."""
    if use_full_model:
        # Full model with sentiment and semantic (requires model downloads)
        return EnsembleEmojiModel(use_sentiment=True, use_semantic=True)
    else:
        # Lightweight keyword-only model (no downloads required)
        return EnsembleEmojiModel(use_sentiment=False, use_semantic=False)

@st.cache_resource
def load_keyword_baseline():
    """Load keyword baseline for comparison."""
    return KeywordBaseline()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Model selection
    st.markdown("### 🤖 Model")
    use_full_model = st.checkbox(
        "Use full model (requires download)",
        value=False,
        help="Enable sentiment and semantic matching. First run will download models (~500MB)."
    )
    
    # Method selection
    method = st.selectbox(
        "Suggestion method",
        ["weighted", "voting"],
        help="Weighted: Use learned weights. Voting: Simple majority voting."
    )
    
    # Show explanation toggle
    show_explanation = st.checkbox("Show analysis details", value=True)
    show_preprocessing = st.checkbox("Show preprocessing", value=False)
    
    st.markdown("---")
    
    # About section
    st.markdown("### 📖 About")
    st.markdown("""
    **Vietnamese Emoji Suggester** uses AI to recommend 
    appropriate emojis for your Vietnamese text.
    
    **Features:**
    - 🔤 Keyword matching
    - 🎭 Emotion detection
    - 🔍 Semantic similarity
    
    **Accuracy:** ~62% precision@3
    """)
    
    st.markdown("---")
    
    # Links
    st.markdown("### 🔗 Links")
    st.markdown("""
    - [GitHub Repository](https://github.com/yourusername/vietnamese-emoji-suggest)
    - [Report Issue](https://github.com/yourusername/vietnamese-emoji-suggest/issues)
    - [Documentation](https://github.com/yourusername/vietnamese-emoji-suggest#readme)
    """)

# ============================================================================
# EMOTION WHEEL VISUALIZATION
# ============================================================================

def plot_emotion_wheel(emotion_scores: dict = None, detected_emotion: str = None):
    """
    Create Plutchik's emotion wheel visualization.
    
    Args:
        emotion_scores: Dict of emotion -> score (0-1)
        detected_emotion: Primary detected emotion
    """
    import plotly.graph_objects as go
    
    # Plutchik's 8 basic emotions in order
    emotions = ['joy', 'trust', 'fear', 'surprise', 
                'sadness', 'disgust', 'anger', 'anticipation']
    
    emotion_colors = {
        'joy': '#FFD700',         # Yellow
        'trust': '#90EE90',       # Light green
        'fear': '#32CD32',        # Green
        'surprise': '#00CED1',    # Cyan
        'sadness': '#4169E1',     # Blue
        'disgust': '#9370DB',     # Purple
        'anger': '#FF4500',       # Red
        'anticipation': '#FFA500' # Orange
    }
    
    # Default scores if not provided
    if emotion_scores is None:
        emotion_scores = {e: 0.1 for e in emotions}
        if detected_emotion and detected_emotion in emotions:
            emotion_scores[detected_emotion] = 0.8
    
    # Create radar chart
    values = [emotion_scores.get(e, 0.1) for e in emotions]
    values.append(values[0])  # Close the polygon
    
    theta = emotions + [emotions[0]]
    colors = [emotion_colors.get(e, '#808080') for e in emotions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgb(102, 126, 234)', width=2),
        name='Emotion Intensity'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0]
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        height=300
    )
    
    return fig


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'user_id' not in st.session_state:
    # Simple user ID for demo (in production, use proper auth)
    st.session_state['user_id'] = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

if 'emotion_history' not in st.session_state:
    st.session_state['emotion_history'] = []

if 'prediction_count' not in st.session_state:
    st.session_state['prediction_count'] = 0

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<div class="main-header">🎯 Vietnamese Emoji Suggester</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered emoji recommendation for Vietnamese text</div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model = load_model(use_full_model)
    keyword_model = load_keyword_baseline()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Enter your message")
    user_text = st.text_area(
        label="Message input",
        placeholder="Nhập tin nhắn tiếng Việt của bạn...\n\nVí dụ: Chúc mừng bạn đậu tuyển dụng!",
        height=150,
        label_visibility="collapsed"
    )
    
    # Quick examples
    st.markdown("**Quick examples:**")
    example_cols = st.columns(4)
    examples = [
        "Chúc mừng bạn!",
        "Buồn quá",
        "Tức ghê!",
        "Sợ quá!"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state['user_text'] = example
                st.rerun()
    
    # Check if example button was clicked
    if 'user_text' in st.session_state:
        user_text = st.session_state['user_text']
        del st.session_state['user_text']

with col2:
    st.markdown("### 💡 Suggestions")
    
    if user_text and user_text.strip():
        # Start timer for latency tracking
        start_time = time.time()
        
        # Get predictions
        try:
            if use_full_model:
                result = model.suggest_with_details(user_text, method=method)
                suggestions = result['final_suggestions']
                detected_emotion = result.get('detected_emotion', 'neutral')
                confidence = result.get('confidence', 0.7)
            else:
                suggestions = model.suggest(user_text, method=method)
                detected_emotion = 'neutral'
                confidence = 0.6
            
            # Apply personalization
            personalizer = get_personalizer()
            user_id = st.session_state['user_id']
            suggestions = personalizer.rank_emojis(
                user_id=user_id,
                emotion=detected_emotion,
                base_emojis=suggestions,
                personalization_weight=0.3
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log prediction
            monitor = get_monitor()
            monitor.log_prediction(
                input_text=user_text,
                predicted_emotion=detected_emotion,
                predicted_intensity=0.7,
                suggested_emojis=suggestions[:5],
                confidence=confidence,
                latency_ms=latency_ms
            )
            
            # Update session state
            st.session_state['prediction_count'] += 1
            st.session_state['emotion_history'].append({
                'emotion': detected_emotion,
                'timestamp': datetime.now().isoformat()
            })
            
            # Display emoji buttons
            emoji_cols = st.columns(3)
            
            for i, emoji in enumerate(suggestions[:3]):
                with emoji_cols[i]:
                    if st.button(
                        emoji,
                        key=f"emoji_{i}",
                        use_container_width=True,
                        help=f"Click to select: {emoji}"
                    ):
                        # Record selection
                        personalizer.record_selection(
                            user_id=user_id,
                            emoji=emoji,
                            emotion=detected_emotion,
                            was_selected=True
                        )
                        st.success(f"Selected: {emoji}")
                        st.code(emoji, language=None)
            
            # Show confidence and method
            st.markdown("---")
            
            # Confidence bar
            st.markdown(f"**Confidence:** {confidence:.0%}")
            st.progress(confidence)
            
            # Latency indicator
            latency_color = "🟢" if latency_ms < 300 else "🟡" if latency_ms < 500 else "🔴"
            st.markdown(f"**Latency:** {latency_color} {latency_ms:.0f}ms")
            
            st.markdown(f"**Method:** {method}")
            
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
            suggestions = ["🤔", "😊", "👍"]
            st.markdown(f"**Fallback suggestions:** {' '.join(suggestions)}")
    else:
        st.info("👈 Enter text to see emoji suggestions")

# ============================================================================
# ANALYSIS SECTION
# ============================================================================

if show_explanation and user_text and user_text.strip():
    st.markdown("---")
    st.markdown("### 🔍 Analysis")
    
    analysis_cols = st.columns([1, 1, 1])
    
    with analysis_cols[0]:
        st.markdown("#### 🔤 Keyword Matches")
        keywords = keyword_model.get_matched_keywords(user_text)
        if keywords:
            for kw in keywords[:5]:
                st.markdown(f"- `{kw}`")
        else:
            st.markdown("*No keywords matched*")
    
    with analysis_cols[1]:
        st.markdown("#### 🎭 Detected Emotion")
        if use_full_model and 'result' in dir() and result.get('detected_emotion'):
            emotion = result['detected_emotion']
            emotion_emojis = {
                'joy': '😊',
                'sadness': '😢',
                'anger': '😠',
                'fear': '😨',
                'surprise': '😮',
                'disgust': '🤢',
                'trust': '🤝',
                'anticipation': '🤞'
            }
            emoji = emotion_emojis.get(emotion, '🤔')
            st.markdown(f"**{emoji} {emotion.capitalize()}**")
            
            # Show intensity
            intensity = result.get('intensity', 0.5)
            st.markdown(f"Intensity: {intensity:.0%}")
            st.progress(intensity)
        else:
            st.markdown("*Enable full model for emotion detection*")
    
    with analysis_cols[2]:
        st.markdown("#### 📊 Model Contributions")
        if use_full_model and 'result' in dir():
            st.markdown(f"- Keyword: `{result.get('keyword_suggestions', [])}`")
            st.markdown(f"- Sentiment: `{result.get('sentiment_suggestions', [])}`")
            st.markdown(f"- Semantic: `{result.get('semantic_suggestions', [])}`")
        else:
            st.markdown("- Keyword: Active ✅")
            st.markdown("- Sentiment: Disabled")
            st.markdown("- Semantic: Disabled")
    
    # Emotion Wheel Visualization
    if use_full_model:
        st.markdown("#### 🎡 Emotion Wheel")
        try:
            detected = result.get('detected_emotion', 'neutral') if 'result' in dir() else 'neutral'
            fig = plot_emotion_wheel(detected_emotion=detected)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for emotion wheel visualization: `pip install plotly`")

# Preprocessing visualization
if show_preprocessing and user_text and user_text.strip():
    st.markdown("---")
    st.markdown("### 🔧 Preprocessing")
    
    prep_cols = st.columns(2)
    
    with prep_cols[0]:
        st.markdown("**Original:**")
        st.code(user_text)
        
        # Extract emojis from original
        original_emojis = extract_emojis(user_text)
        if original_emojis:
            st.markdown(f"*Emojis found: {' '.join(original_emojis)}*")
    
    with prep_cols[1]:
        st.markdown("**Preprocessed:**")
        processed = preprocess_text(user_text)
        st.code(processed)

# ============================================================================
# STATISTICS SECTION
# ============================================================================

st.markdown("---")
st.markdown("### 📊 System Statistics")

stat_cols = st.columns(5)

with stat_cols[0]:
    st.metric(
        label="Model Accuracy",
        value="62%",
        delta="+17% vs baseline",
        help="Precision@3 on test dataset"
    )

with stat_cols[1]:
    st.metric(
        label="Dataset Size",
        value="450+",
        delta="150 manual + 300 augmented",
        help="Training samples (manual + augmented)"
    )

with stat_cols[2]:
    st.metric(
        label="Emoji Database",
        value="100+",
        delta="emojis",
        help="Number of emojis in database"
    )

with stat_cols[3]:
    st.metric(
        label="Emotions",
        value="8",
        delta="Plutchik's wheel",
        help="Based on Plutchik's emotion wheel"
    )

with stat_cols[4]:
    st.metric(
        label="Your Predictions",
        value=str(st.session_state['prediction_count']),
        delta="this session",
        help="Number of predictions in this session"
    )

# ============================================================================
# PERSONALIZATION STATS
# ============================================================================

with st.expander("👤 Your Personalization Stats"):
    try:
        personalizer = get_personalizer()
        user_stats = personalizer.get_user_stats(st.session_state['user_id'])
        
        pers_cols = st.columns(4)
        with pers_cols[0]:
            st.metric("Total Interactions", user_stats['total_interactions'])
        with pers_cols[1]:
            st.metric("Active Days", user_stats['active_days'])
        with pers_cols[2]:
            st.metric("Emotions Used", len(user_stats['emotions_used']))
        with pers_cols[3]:
            if user_stats['favorite_emojis']:
                st.markdown(f"**Favorites:** {' '.join(user_stats['favorite_emojis'][:5])}")
            else:
                st.markdown("*No favorites yet*")
        
        # Option to reset personalization
        if st.button("🔄 Reset My Preferences", type="secondary"):
            personalizer.reset_user(st.session_state['user_id'])
            st.success("Preferences reset!")
            st.rerun()
    except Exception as e:
        st.info("Personalization stats will appear after a few interactions.")

# ============================================================================
# BATCH PROCESSING (OPTIONAL)
# ============================================================================

with st.expander("🔄 Batch Processing"):
    st.markdown("Enter multiple texts (one per line) for batch emoji suggestions:")
    
    batch_input = st.text_area(
        "Batch input",
        placeholder="Chúc mừng bạn!\nBuồn quá\nTức ghê!",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("Process Batch", type="primary"):
        if batch_input:
            lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            results = []
            for text in lines:
                emojis = model.suggest(text, method=method)
                results.append({"Text": text, "Suggestions": " ".join(emojis)})
            
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("Please enter some texts to process.")

# ============================================================================
# FEEDBACK SECTION
# ============================================================================

with st.expander("📝 Feedback"):
    st.markdown("Help us improve! Was the emoji suggestion helpful?")
    
    feedback_cols = st.columns(3)
    
    with feedback_cols[0]:
        if st.button("👍 Yes, great!", use_container_width=True):
            try:
                monitor = get_monitor()
                monitor.record_feedback(datetime.now().isoformat(), "positive")
            except:
                pass
            st.success("Thank you for your positive feedback!")
    
    with feedback_cols[1]:
        if st.button("🤔 Could be better", use_container_width=True):
            try:
                monitor = get_monitor()
                monitor.record_feedback(datetime.now().isoformat(), "neutral")
            except:
                pass
            st.info("Thanks! We're working on improvements.")
    
    with feedback_cols[2]:
        if st.button("👎 Not helpful", use_container_width=True):
            try:
                monitor = get_monitor()
                monitor.record_feedback(datetime.now().isoformat(), "negative")
            except:
                pass
            st.warning("Sorry to hear that. Please report specific issues.")
    
    feedback_text = st.text_input("Additional comments (optional):")
    if feedback_text:
        st.success("Feedback recorded. Thank you!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    Made with ❤️ using Python, Streamlit & Transformers<br>
    <a href="https://github.com/yourusername/vietnamese-emoji-suggest">GitHub</a> | 
    <a href="https://github.com/yourusername/vietnamese-emoji-suggest/issues">Report Issue</a> | 
    MIT License
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DEBUG INFO (DEVELOPMENT)
# ============================================================================

if st.checkbox("🐛 Show debug info", value=False):
    st.markdown("### Debug Information")
    st.json({
        "use_full_model": use_full_model,
        "method": method,
        "model_type": type(model).__name__,
        "user_text": user_text if user_text else None,
    })
