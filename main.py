import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from src.preprocessing import Connect4Preprocessor

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_API")

app = FastAPI(title="Connect4 ML Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("/app/models")

# ------------------------------------------------------------------
# Global State (SEPARATE MODELS)
# ------------------------------------------------------------------

# Policy model
policy_model = None
policy_preprocessor = None
policy_version = None

# Win probability model
winprob_model = None
winprob_preprocessor = None
winprob_version = None

# ------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------

class BoardState(BaseModel):
    board: List[List[int]]  # 6x7
    current_player: int
    legal_moves: List[int]


class PredictionRequest(BaseModel):
    board_state: BoardState
    top_k: Optional[int] = 3


class WinProbRequest(BaseModel):
    board_before: List[List[int]]  # 6x7
    policy: List[float]            # len=7
    q_values: List[float]          # len=7
    move_index: int


class DeployRequest(BaseModel):
    version: str
    model_type: str  # "policy" | "winprob"

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_policy_artifacts(version: str) -> bool:
    global policy_model, policy_preprocessor, policy_version

    model_path = MODEL_DIR / f"model_{version}.joblib"
    prep_path = MODEL_DIR / f"preprocessor_{version}.joblib"

    if not model_path.exists() or not prep_path.exists():
        logger.error(f"Policy artifacts missing for {version}")
        return False

    policy_model = joblib.load(model_path)
    policy_preprocessor = joblib.load(prep_path)
    policy_version = version

    logger.info(f"Policy model {version} loaded")
    return True


def load_winprob_artifacts(version: str) -> bool:
    global winprob_model, winprob_preprocessor, winprob_version

    model_path = MODEL_DIR / f"winprob_model_{version}.joblib"
    prep_path = MODEL_DIR / f"winprob_preprocessor_{version}.joblib"

    if not model_path.exists() or not prep_path.exists():
        logger.error(f"WinProb artifacts missing for {version}")
        return False

    winprob_model = joblib.load(model_path)
    winprob_preprocessor = joblib.load(prep_path)
    winprob_version = version

    logger.info(f"WinProb model {version} loaded")
    return True

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def preprocess_board_for_policy(board: List[List[int]]) -> np.ndarray:
    flat_board = np.array(board).flatten()
    feature_names = [f"board_{i}" for i in range(42)]
    df = pd.DataFrame([flat_board], columns=feature_names)
    return df.values

# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("ML API starting up...")

    if not MODEL_DIR.exists():
        logger.warning("Model directory missing")
        return

    # Auto-load latest policy model if present
    policy_files = sorted(MODEL_DIR.glob("model_*.joblib"))
    if policy_files:
        version = policy_files[-1].stem.split("_")[-1]
        load_policy_artifacts(version)

    # Auto-load latest winprob model if present
    winprob_files = sorted(MODEL_DIR.glob("winprob_model_*.joblib"))
    if winprob_files:
        version = winprob_files[-1].stem.split("_")[-1]
        load_winprob_artifacts(version)

# ------------------------------------------------------------------
# Deploy
# ------------------------------------------------------------------

@app.post("/deploy")
async def deploy_model(request: DeployRequest):
    logger.info(f"Deploy request: {request.model_type} {request.version}")

    if request.model_type == "policy":
        if not load_policy_artifacts(request.version):
            raise HTTPException(500, "Failed to load policy model")

    elif request.model_type == "winprob":
        if not load_winprob_artifacts(request.version):
            raise HTTPException(500, "Failed to load win probability model")

    else:
        raise HTTPException(400, "Unknown model_type")

    return {
        "status": "deployed",
        "model_type": request.model_type,
        "version": request.version,
    }

# ------------------------------------------------------------------
# Policy Prediction
# ------------------------------------------------------------------

@app.post("/predict")
async def predict_move(request: PredictionRequest):
    if policy_model is None:
        raise HTTPException(503, "Policy model not loaded")

    X = preprocess_board_for_policy(request.board_state.board)
    probs = policy_model.predict_proba(X)[0]

    legal_moves = set(request.board_state.legal_moves)
    legal_probs = {i: float(probs[i]) for i in legal_moves}

    if not legal_probs:
        raise HTTPException(400, "No legal moves")

    best_move = max(legal_probs, key=legal_probs.get)

    sorted_moves = sorted(
        legal_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "predicted_move": int(best_move),
        "confidence": legal_probs[best_move],
        "top_k_moves": [
            {"move": m, "confidence": c}
            for m, c in sorted_moves[:request.top_k]
        ],
        "model_version": policy_version,
    }

# ------------------------------------------------------------------
# Win Probability Prediction
# ------------------------------------------------------------------

@app.post("/predict/win-probability")
async def predict_win_probability(req: WinProbRequest):
    if winprob_model is None or winprob_preprocessor is None:
        raise HTTPException(503, "Win probability model not loaded")

    row = {}

    # Board
    for r in range(6):
        for c in range(7):
            row[f"board_before_r{r}c{c}"] = req.board_before[r][c]

    # Policy + Q
    for i in range(7):
        row[f"policy_col_{i}"] = req.policy[i]
        row[f"q_value_col_{i}"] = req.q_values[i]

    row["moveIndex"] = req.move_index
    row["gameId"] = "inference"
    row["game_outcome"] = "UNKNOWN"
    row["game_winner"] = "UNKNOWN"

    df = pd.DataFrame([row])

    X = winprob_preprocessor.transform_new_data(df)
    raw_proba = winprob_model.predict_proba(X)[0]

    # Normalize to [LOSS, DRAW, WIN]
    proba3 = np.zeros(3, dtype=float)

    if len(raw_proba) == 3:
        proba3[:] = raw_proba

    elif len(raw_proba) == 2:
        # Binary model: need to infer which class is missing
        classes = winprob_model.classes_

        for i, cls in enumerate(classes):
            proba3[int(cls)] = raw_proba[i]

    else:
        raise RuntimeError(
            f"Unexpected proba shape: {raw_proba.shape}"
        )

    return {
        "loss": float(proba3[0]),
        "draw": float(proba3[1]),
        "win": float(proba3[2]),
        "model_version": winprob_version,
    }


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "policy_loaded": policy_model is not None,
        "policy_version": policy_version,
        "winprob_loaded": winprob_model is not None,
        "winprob_version": winprob_version,
    }
