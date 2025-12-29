import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
from src.preprocessing import Connect4Preprocessor

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_API")

app = FastAPI(title="Connect4 ML Inference API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
MODEL_DIR = Path("/app/models")
current_model = None
current_preprocessor = None
current_version = "v0"


# --- Data Models ---

class BoardState(BaseModel):
    board: List[List[int]]  # 6x7 matrix
    current_player: int
    legal_moves: List[int]


class PredictionRequest(BaseModel):
    board_state: BoardState
    top_k: Optional[int] = 3


class DeployRequest(BaseModel):
    version: str


# --- Helper Functions ---

def load_model_artifacts(version: str):
    """Loads model and preprocessor for a specific version."""
    global current_model, current_preprocessor, current_version

    model_path = MODEL_DIR / f"model_{version}.joblib"
    prep_path = MODEL_DIR / f"preprocessor_{version}.joblib"

    if not model_path.exists() or not prep_path.exists():
        logger.error(f" Model files not found for version {version}")
        return False

    try:
        logger.info(f" Loading version {version}...")
        current_model = joblib.load(model_path)
        current_preprocessor = joblib.load(prep_path)
        current_version = version
        logger.info(f" Successfully loaded version {version}")
        return True
    except Exception as e:
        logger.error(f" Failed to load artifacts: {e}")
        return False


def preprocess_board_for_inference(board: List[List[int]]) -> np.ndarray:
    """
    Convert board to scaled features using the preprocessor.
    This fixes the sklearn warning and ensures proper scaling.
    """
    # Flatten the board to 42 features
    flat_board = np.array(board).flatten()

    #  Create feature names matching training (board_0, board_1, ..., board_41)
    feature_names = [f"board_{i}" for i in range(42)]

    #   Create DataFrame with proper feature names
    # The scaler was fitted on data with these column names during training
    features_df = pd.DataFrame([flat_board], columns=feature_names)

    scaled_features = features_df.values

    # Log for debugging (remove in production)
    logger.debug(f" Board (first 10): {flat_board[:10]}")
    logger.debug(f" Scaled (first 10): {scaled_features[0][:10]}")

    return scaled_features


# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Try to load the latest model on startup."""
    logger.info(" ML API starting up...")

    if not MODEL_DIR.exists():
        logger.warning(f" Model directory {MODEL_DIR} does not exist.")
        return

    # Find all model files
    files = list(MODEL_DIR.glob("model_*.joblib"))
    if not files:
        logger.warning(" No models found in /app/models. Waiting for training...")
        return

    # Sort by version
    latest_model = sorted(files)[-1]
    try:
        version = latest_model.stem.split("_")[-1]  # Extract 'v1' from 'model_v1'
        logger.info(f" Found latest model: {version}")
        load_model_artifacts(version)
    except Exception as e:
        logger.error(f"Error loading model on startup: {e}")


@app.post("/deploy")
async def deploy_model(request: DeployRequest, background_tasks: BackgroundTasks):
    """
    Endpoint called by the Training Service to trigger a reload.
    """
    version = request.version
    logger.info(f" Received deployment signal for {version}")

    # Check if files exist
    if not (MODEL_DIR / f"model_{version}.joblib").exists():
        raise HTTPException(status_code=404, detail="Model version files not found")

    # Load immediately
    success = load_model_artifacts(version)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")

    return {"status": "deployed", "version": version}


@app.post("/predict")
async def predict_move(request: PredictionRequest):
    """
    Main inference endpoint.
    """
    if current_model is None:
        raise HTTPException(status_code=503, detail="No model loaded yet")

    try:
        # Use the corrected preprocessing function
        scaled_features = preprocess_board_for_inference(request.board_state.board)

        # 2. Predict Probabilities
        probs = current_model.predict_proba(scaled_features)[0]

        # 3. Filter Illegal Moves
        legal_moves = set(request.board_state.legal_moves)
        legal_probs = {}

        for col in range(7):
            if col in legal_moves:
                legal_probs[col] = float(probs[col])

        if not legal_probs:
            raise HTTPException(status_code=400, detail="No legal moves available")

        # 4. Select Best Move
        best_move = max(legal_probs, key=legal_probs.get)
        confidence = legal_probs[best_move]

        # 5. Top K Moves
        sorted_moves = sorted(
            legal_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "predicted_move": int(best_move),
            "confidence": confidence,
            "top_k_moves": [
                {"move": m, "confidence": c} for m, c in sorted_moves[:request.top_k]
            ],
            "model_version": current_version
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if current_model is not None else "waiting",
        "model_loaded": current_model is not None,
        "current_version": current_version,
        "model_dir": str(MODEL_DIR)
    }


@app.get("/debug/test-prediction")
async def debug_test_prediction():
    """
    Debug endpoint to test if predictions change with different boards.
    """
    if current_model is None:
        return {"error": "No model loaded"}

    # Test 1: Empty board
    empty_board = [[0]*7 for _ in range(6)]
    empty_features = preprocess_board_for_inference(empty_board)
    empty_probs = current_model.predict_proba(empty_features)[0]

    # Test 2: Board with one piece
    one_piece_board = [[0]*7 for _ in range(6)]
    one_piece_board[5][3] = 1
    one_piece_features = preprocess_board_for_inference(one_piece_board)
    one_piece_probs = current_model.predict_proba(one_piece_features)[0]

    return {
        "empty_board_probs": empty_probs.tolist(),
        "one_piece_board_probs": one_piece_probs.tolist(),
        "are_different": not np.allclose(empty_probs, one_piece_probs),
        "message": "Predictions should be DIFFERENT if model is working correctly"
    }