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
MODELS = {
    "policy": {
        "model": None,
        "preprocessor": None,
        "version": None,
    },
    "winprob": {
        "model": None,
        "preprocessor": None,
        "version": None,
    }
}

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
    """
    Loads BOTH policy and win-prob models if they exist.
    Does NOT fail if one is missing.
    """
    loaded_any = False

    # -------------------------
    # POLICY MODEL (existing)
    # -------------------------
    policy_model_path = MODEL_DIR / f"model_{version}.joblib"
    policy_prep_path = MODEL_DIR / f"preprocessor_{version}.joblib"

    if policy_model_path.exists() and policy_prep_path.exists():
        try:
            MODELS["policy"]["model"] = joblib.load(policy_model_path)
            MODELS["policy"]["preprocessor"] = joblib.load(policy_prep_path)
            MODELS["policy"]["version"] = version
            logger.info(f" Policy model loaded: {version}")
            loaded_any = True
        except Exception as e:
            logger.error(f" Failed to load policy model: {e}")

    # -------------------------
    # WIN-PROB MODEL (new)
    # -------------------------
    win_model_path = MODEL_DIR / f"winprob_model_{version}.joblib"
    win_prep_path = MODEL_DIR / f"winprob_preprocessor_{version}.joblib"

    if win_model_path.exists() and win_prep_path.exists():
        try:
            MODELS["winprob"]["model"] = joblib.load(win_model_path)
            MODELS["winprob"]["preprocessor"] = joblib.load(win_prep_path)
            MODELS["winprob"]["version"] = version
            logger.info(f" WinProb model loaded: {version}")
            loaded_any = True
        except Exception as e:
            logger.error(f" Failed to load win-prob model: {e}")

    return loaded_any


def is_winprob_model(model) -> bool:
    """
    Win-prob models output 3-class probabilities:
    [LOSS, DRAW, WIN]
    """
    try:
        n_classes = model.n_classes_
        return n_classes == 3
    except AttributeError:
        return False

def align_board_to_player(board: list[list[int]], current_player: int) -> np.ndarray:
    """
    Align board so that the model always sees the position
    from the perspective of Player 1.
    """
    board = np.array(board, dtype=int)

    if current_player == 2:
        # swap 1 <-> 2
        board = np.where(board == 1, 2, np.where(board == 2, 1, board))

    return board.flatten().reshape(1, -1)



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
    version = request.version
    logger.info(f"Received deployment signal for {version}")

    if not (MODEL_DIR / f"model_{version}.joblib").exists():
        raise HTTPException(status_code=404, detail="Model version files not found")

    success = load_model_artifacts(version)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load model")

    return {"status": "deployed", "version": version}



@app.post("/predict")
async def predict_move(request: PredictionRequest):
    model = MODELS["policy"]["model"]

    if model is None:
        raise HTTPException(status_code=503, detail="Policy model not loaded")

    scaled_features = preprocess_board_for_inference(
        request.board_state.board
    )

    probs = model.predict_proba(scaled_features)[0]

    legal_moves = set(request.board_state.legal_moves)
    legal_probs = {
        col: float(probs[col])
        for col in legal_moves
    }

    best_move = max(legal_probs, key=legal_probs.get)

    return {
        "predicted_move": int(best_move),
        "confidence": legal_probs[best_move],
        "top_k_moves": [
            {"move": m, "confidence": c}
            for m, c in sorted(
                legal_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:request.top_k]
        ],
        "model_version": MODELS["policy"]["version"]
    }

@app.post("/predict-winprob")
async def predict_winprob(request: PredictionRequest):
    model = MODELS["winprob"]["model"]

    if model is None:
        raise HTTPException(status_code=503, detail="Win-prob model not loaded")

    try:
        # 1) Align board to side-to-move
        aligned_board = align_board_to_player(
            request.board_state.board,
            request.board_state.current_player
        )

        # 2) Build FULL 65-feature vector
        X = Connect4Preprocessor.winprob_features_from_board(aligned_board)

        # 3) Predict
        probs = model.predict_proba(X)[0]

        if len(probs) == 2:
            loss_p, win_p = probs
            draw_p = 0.0
            model_type = "binary"
        else:
            loss_p, draw_p, win_p = probs
            model_type = "multiclass"

        return {
            "loss": float(loss_p),
            "draw": float(draw_p),
            "win": float(win_p),
            "model_type": model_type,
            "model_version": MODELS["winprob"]["version"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "policy_loaded": MODELS["policy"]["model"] is not None,
        "policy_version": MODELS["policy"]["version"],
        "winprob_loaded": MODELS["winprob"]["model"] is not None,
        "winprob_version": MODELS["winprob"]["version"],
    }



@app.get("/debug/test-prediction")
async def debug_test_prediction(model_type: str = "policy"):
    """
    Debug endpoint to verify model sensitivity to board changes.

    model_type:
      - "policy"
      - "winprob"
    """
    if model_type not in MODELS:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'policy' or 'winprob'"
        )

    model = MODELS[model_type]["model"]

    if model is None:
        return {"error": f"{model_type} model not loaded"}

    # --- Test boards ---
    empty_board = [[0] * 7 for _ in range(6)]
    one_piece_board = [[0] * 7 for _ in range(6)]
    one_piece_board[5][3] = 1

    # --- Feature prep ---
    X_empty = preprocess_board_for_inference(empty_board)
    X_one = preprocess_board_for_inference(one_piece_board)

    # --- Predict ---
    empty_probs = model.predict_proba(X_empty)[0]
    one_piece_probs = model.predict_proba(X_one)[0]

    return {
        "model_type": model_type,
        "model_version": MODELS[model_type]["version"],
        "empty_board_probs": empty_probs.tolist(),
        "one_piece_board_probs": one_piece_probs.tolist(),
        "are_different": not np.allclose(empty_probs, one_piece_probs),
        "message": "Predictions should be DIFFERENT if the model is working correctly"
    }
