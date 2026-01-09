"""
FastAPI backend for Vietnamese Emoji Suggestion System

Run with: uvicorn app.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EnsembleEmojiModel, KeywordBaseline
from src.preprocessing import preprocess_text

# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Vietnamese Emoji Suggestion API",
    description="AI-powered emoji recommendation for Vietnamese text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

# Load models at startup
keyword_model = KeywordBaseline()
ensemble_model = None  # Lazy loaded

def get_ensemble_model():
    """Lazy load ensemble model."""
    global ensemble_model
    if ensemble_model is None:
        ensemble_model = EnsembleEmojiModel(use_sentiment=False, use_semantic=False)
    return ensemble_model

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class SuggestionRequest(BaseModel):
    """Request body for emoji suggestion."""
    text: str = Field(..., description="Vietnamese text to analyze", min_length=1, max_length=500)
    method: str = Field(default="weighted", description="Suggestion method: 'voting' or 'weighted'")
    top_k: int = Field(default=3, description="Number of suggestions to return", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Chúc mừng bạn đậu tuyển dụng!",
                "method": "weighted",
                "top_k": 3
            }
        }

class SuggestionResponse(BaseModel):
    """Response body for emoji suggestion."""
    text: str = Field(..., description="Original input text")
    suggestions: List[str] = Field(..., description="List of suggested emojis")
    method: str = Field(..., description="Method used for suggestion")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Chúc mừng bạn đậu tuyển dụng!",
                "suggestions": ["😊", "🎉", "🥳"],
                "method": "weighted"
            }
        }

class DetailedSuggestionResponse(BaseModel):
    """Detailed response with analysis breakdown."""
    text: str
    preprocessed_text: str
    suggestions: List[str]
    keyword_suggestions: List[str]
    detected_emotion: Optional[str]
    matched_keywords: List[str]
    method: str

class BatchRequest(BaseModel):
    """Request for batch processing."""
    texts: List[str] = Field(..., description="List of texts to process", min_items=1, max_items=100)
    method: str = Field(default="weighted")

class BatchResponse(BaseModel):
    """Response for batch processing."""
    results: List[Dict[str, Any]]
    count: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Vietnamese Emoji Suggestion API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "keyword_model": keyword_model is not None,
            "ensemble_model": ensemble_model is not None
        }
    )

@app.post("/suggest", response_model=SuggestionResponse, tags=["Suggestions"])
async def suggest_emoji(request: SuggestionRequest):
    """
    Suggest emojis for Vietnamese text.
    
    - **text**: Vietnamese text to analyze (required)
    - **method**: 'weighted' or 'voting' (default: weighted)
    - **top_k**: Number of suggestions (default: 3)
    """
    try:
        model = get_ensemble_model()
        suggestions = model.suggest(request.text, method=request.method)
        
        return SuggestionResponse(
            text=request.text,
            suggestions=suggestions[:request.top_k],
            method=request.method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest/detailed", response_model=DetailedSuggestionResponse, tags=["Suggestions"])
async def suggest_emoji_detailed(request: SuggestionRequest):
    """
    Get detailed emoji suggestions with analysis breakdown.
    """
    try:
        model = get_ensemble_model()
        
        # Get detailed results
        result = model.suggest_with_details(request.text, method=request.method)
        
        # Preprocess text
        preprocessed = preprocess_text(request.text)
        
        return DetailedSuggestionResponse(
            text=request.text,
            preprocessed_text=preprocessed,
            suggestions=result['final_suggestions'][:request.top_k],
            keyword_suggestions=result.get('keyword_suggestions', []),
            detected_emotion=result.get('detected_emotion'),
            matched_keywords=result.get('matched_keywords', []),
            method=request.method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest/batch", response_model=BatchResponse, tags=["Suggestions"])
async def suggest_emoji_batch(request: BatchRequest):
    """
    Batch process multiple texts for emoji suggestions.
    """
    try:
        model = get_ensemble_model()
        
        results = []
        for text in request.texts:
            suggestions = model.suggest(text, method=request.method)
            results.append({
                "text": text,
                "suggestions": suggestions
            })
        
        return BatchResponse(
            results=results,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess", tags=["Utilities"])
async def preprocess_endpoint(text: str):
    """
    Preprocess Vietnamese text (normalize teencode, remove URLs, etc.).
    """
    try:
        processed = preprocess_text(text)
        return {
            "original": text,
            "processed": processed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions", tags=["Info"])
async def get_emotions():
    """
    Get list of supported emotions and their emoji mappings.
    """
    from src.models import EMOTION_EMOJI_MAP
    return {
        "emotions": list(EMOTION_EMOJI_MAP.keys()),
        "mappings": EMOTION_EMOJI_MAP
    }

@app.get("/stats", tags=["Info"])
async def get_stats():
    """
    Get model statistics and information.
    """
    from src.models import EMOJI_DESCRIPTIONS
    return {
        "model_version": "1.0.0",
        "num_emojis": len(EMOJI_DESCRIPTIONS),
        "supported_methods": ["voting", "weighted"],
        "accuracy": {
            "precision_at_3": 0.62,
            "baseline": 0.45
        }
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
