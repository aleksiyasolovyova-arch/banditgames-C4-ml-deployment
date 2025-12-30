# Connect4 ML Deployment Service

This directory contains the **deployment and inference layer** for the Connect4 machine learning system.
It exposes trained models through a **FastAPI service**, handling preprocessing, versioning, and safe inference
for both **policy** and **win-probability** models.

---

##  Purpose

The deployment service is responsible for:
- Loading trained models and preprocessors
- Exposing HTTP endpoints for inference
- Ensuring training–inference feature consistency
- Supporting hot model deployment without downtime
- Serving both move-policy and win-probability predictions

---

##  Supported Models

### 1. Policy Model
- Predicts a probability distribution over the 7 possible columns
- Used to select the next move
- Output is filtered to **legal moves only**

### 2. Win Probability Model
- Predicts probabilities for:
  - LOSS
  - DRAW
  - WIN
- Always evaluated from the **current player’s perspective**
- Returns calibrated probabilities for downstream decision-making

---

##  Deployment Preprocessing

### Board Representation

The deployment layer accepts boards as:
```json
{
  "board": [[0,0,0,0,0,0,0], ...],
  "current_player": 1,
  "legal_moves": [0,1,2,3,4,5,6]
}
```

Internally:
- Boards are flattened into 42 features
- Player perspective is normalized so the model always sees **Player 1**
- Feature ordering matches training exactly

This guarantees **training–inference parity**.

---

##  Perspective Alignment (Critical)

Before inference, the board is aligned so that:
- Player 1 is always the side to move
- If `current_player == 2`, all pieces are swapped (1 ↔ 2)

This allows a **single model** to generalize across both players without retraining.

---

##  Win-Probability Feature Construction

For win-probability inference, the service reconstructs the **exact 65-feature vector** used in training:
- 42 board cells
- 7 policy placeholders
- 7 Q-value placeholders
- 9 engineered features

This strict feature matching prevents silent inference bugs.

---

##  Model Versioning & Deployment

### Model Storage
Models are loaded from:
```
/app/models/
```

Artifacts include:
- `model_<version>.joblib`
- `preprocessor_<version>.joblib`
- `winprob_model_<version>.joblib`
- `winprob_preprocessor_<version>.joblib`

---

### Hot Deployment Endpoint

`POST /deploy`

Allows switching models **at runtime** without restarting the service.

---

##  Inference Endpoints

### Policy Prediction
`POST /predict`

Returns:
- Best legal move
- Confidence score
- Top-K alternatives
- Active model version

---

### Win Probability Prediction
`POST /predict-winprob`

Returns calibrated probabilities for LOSS, DRAW, and WIN.

---

## 🩺 Health & Debugging

- `GET /health` – model status

---

---

## 🧠 Summary

This service is the production interface between trained Connect4 models and downstream consumers.
It ensures correctness, stability, and trust in both move selection and win-probability outputs.
