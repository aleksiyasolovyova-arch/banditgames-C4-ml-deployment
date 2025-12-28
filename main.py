"""
Connect4 ML API - Inference Microservice

Serves move predictions via REST API.
Fast, lightweight, stateless inference service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
from pathlib import Path
import logging
import os
from datetime import datetime

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="Connect4 ML API",
    description="Policy imitation model for Connect4 move prediction",
    version="1.0.0"
)

# Models (loaded on startup)
model = None
preprocessor = None
model_version = None
model_loaded_at = None


class BoardState(BaseModel):
    """Board state representation"""
    board: List[List[int]] = Field(..., description="6x7 board (0=empty, 1=player1, 2=player2)")
    current_player: int = Field(..., description="Current player (1 or 2)")
    legal_moves: List[int] = Field(..., description="Legal columns [0-6]")


class PredictionRequest(BaseModel):
    """Prediction request"""
    board_state: BoardState
    top_k: Optional[int] = Field(3, description="Number of top moves to return")


class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_move: int
    confidence: float
    top_k_moves: List[Dict[str, float]]
    inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    model_loaded_at: Optional[str]
    uptime_seconds: float


# Startup
@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_version, model_loaded_at
    
    logger.info("Loading model...")
    
    try:
        # Paths from environment
        model_path = os.getenv('MODEL_PATH', '/app/models/v1/xgboost/xgboost_model_v1.joblib')
        preprocessor_path = os.getenv('PREPROCESSOR_PATH', '/app/models/v1/preprocessing/preprocessor.joblib')
        model_version = os.getenv('MODEL_VERSION', 'v1')
        
        # Load
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        model_loaded_at = datetime.now()
        
        logger.info(f"Model loaded successfully: {model_version}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Preprocessor path: {preprocessor_path}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_board(board_state: BoardState) -> np.ndarray:
    """
    Preprocess board state for prediction.
    
    Args:
        board_state: Board state
        
    Returns:
        Feature vector
    """
    # Convert board to numpy array
    board = np.array(board_state.board, dtype=np.int8)
    
    # Extract features (simplified - full preprocessing in actual model)
    features = []
    
    # Flatten board
    features.extend(board.flatten())
    
    # Current player
    features.append(board_state.current_player)
    
    # Legal moves (binary)
    legal_moves_binary = [1 if i in board_state.legal_moves else 0 for i in range(7)]
    features.extend(legal_moves_binary)
    
    return np.array([features])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - model_loaded_at).total_seconds() if model_loaded_at else 0
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        model_loaded_at=model_loaded_at.isoformat() if model_loaded_at else None,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_move(request: PredictionRequest):
    """
    Predict best move for given board state.
    
    Args:
        request: Prediction request with board state
        
    Returns:
        Predicted move with confidence and top-k alternatives
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Preprocess
        features = preprocess_board(request.board_state)
        
        # Predict probabilities
        probabilities = model.predict_proba(features)[0]
        
        # Filter to legal moves only
        legal_probs = {
            move: probabilities[move] 
            for move in request.board_state.legal_moves
        }
        
        # Sort by probability
        sorted_moves = sorted(legal_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Best move
        best_move, best_confidence = sorted_moves[0]
        
        # Top-k moves
        top_k_moves = [
            {"move": int(move), "confidence": float(conf)}
            for move, conf in sorted_moves[:request.top_k]
        ]
        
        # Inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predicted_move=int(best_move),
            confidence=float(best_confidence),
            top_k_moves=top_k_moves,
            inference_time_ms=inference_time,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Connect4 ML API",
        "version": "1.0.0",
        "model_version": model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
