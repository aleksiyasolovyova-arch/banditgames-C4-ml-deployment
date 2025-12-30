
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List
from pathlib import Path
import joblib


logger = logging.getLogger(__name__)


class Connect4Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [f"board_{i}" for i in range(42)]

    def _parse_board_string(self, series: pd.Series) -> pd.DataFrame:
        """Parses '.......X...O' string into 42 numerical features"""
        char_map = {'.': 0, 'X': 1, 'O': 2, '1': 1, '2': 2}

        def parse(s):
            if not isinstance(s, str): return [0] * 42
            s = s.strip().replace('\n', '')[:42]
            return [char_map.get(c, 0) for c in s]

        return pd.DataFrame(series.apply(parse).tolist(), columns=self.feature_columns)

    def preprocess_pipeline(self, dataset_path: str) -> Dict[str, np.ndarray]:
        logger.info(f"📥 Loading dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path)

        # --- LEAKAGE PREVENTION START ---
        # 1. Identify unique Games to split by Game ID, not move
        if 'gameId' in df.columns:
            unique_games = df['gameId'].unique()
            np.random.seed(42)
            np.random.shuffle(unique_games)

            # Split Game IDs (80% Train, 20% Test)
            split_idx = int(len(unique_games) * 0.8)
            train_games = set(unique_games[:split_idx])
            test_games = set(unique_games[split_idx:])

            # Filter DataFrame
            train_df = df[df['gameId'].isin(train_games)].copy()
            test_df = df[df['gameId'].isin(test_games)].copy()
            logger.info(f"Split by GameID: {len(train_games)} Train games, {len(test_games)} Test games")
        else:
            logger.warning(" 'gameId' column missing! Falling back to random split (Potential Leakage).")
            # Fallback logic (simple split)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

        # --- LEAKAGE PREVENTION END ---

        # 2. Helper to Parse Features
        def get_X_y(subset_df):
            if 'board_before' in subset_df.columns:
                X = self._parse_board_string(subset_df['board_before'])
            else:
                # Fallback to existing columns if strings aren't present
                cols = [c for c in subset_df.columns if 'board_before' in c]
                X = subset_df[cols].fillna(0)
                # Ensure standard column names
                X.columns = self.feature_columns[:len(X.columns)]

            y = None
            if 'actionTaken' in subset_df.columns:  # ← FIX: Use camelCase
                y = subset_df['actionTaken'].astype(int)
            elif 'action_taken' in subset_df.columns:  # Fallback for old datasets
                y = subset_df['action_taken'].astype(int)
            elif 'moveIndex' in subset_df.columns:
                y = subset_df['moveIndex'].astype(int)

            return X, y

        X_train, y_train = get_X_y(train_df)
        X_test, y_test = get_X_y(test_df)

        # 3. Scale (Fit on TRAIN only, Transform TEST)
        X_train_scaled = X_train
        X_test_scaled = X_test

        return {
            "X_train": X_train_scaled, "y_train": y_train,
            "X_test": X_test_scaled, "y_test": y_test,
            "feature_names": self.feature_columns
        }


# ============================================================
# WIN PROBABILITY PREPROCESSOR
# (MUST live in this file for pickle compatibility)
# ============================================================

class Connect4WinProbPreprocessor:
    def __init__(self, self_play: bool = True):
        self.self_play = self_play
        self.scaler = StandardScaler()
        self.feature_columns: List[str] | None = None

    # ------------------------------------------------------------------
    # Perspective augmentation (SELF-PLAY)
    # ------------------------------------------------------------------
    def augment_selfplay_perspective(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["perspective"] = "original"

        mirror = df.copy()
        mirror["perspective"] = "mirrored"

        board_before = [c for c in df.columns if c.startswith("board_before_r")]

        arr = mirror[board_before].to_numpy()
        arr = np.where(arr == 1, 2, np.where(arr == 2, 1, arr))
        mirror[board_before] = arr

        return pd.concat([df, mirror], ignore_index=True)

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    def build_outcome_target(self, df: pd.DataFrame) -> pd.Series:
        y = pd.Series(0, index=df.index, dtype=int)

        outcome = df["game_outcome"].astype(str).str.upper().str.strip()
        winner = df["game_winner"].astype(str).str.lower().str.strip()

        y[outcome == "DRAW"] = 1

        if self.self_play:
            non_draw = outcome != "DRAW"
            y[non_draw & (df["perspective"] == "original")] = 0
            y[non_draw & (df["perspective"] == "mirrored")] = 2
        else:
            non_draw = outcome != "DRAW"
            y[non_draw & winner.str.startswith("ai")] = 2

        return y

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["gameId", "moveIndex", "game_outcome", "game_winner"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        fill_cols = [c for c in df.columns if c.startswith(("policy_col_", "q_value_col_"))]
        df[fill_cols] = df[fill_cols].fillna(0)

        return df.dropna(subset=["moveIndex"])

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        board_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in range(7)]
        board = df[board_cols].to_numpy()

        df["player1_pieces"] = (board == 1).sum(axis=1)
        df["player2_pieces"] = (board == 2).sum(axis=1)

        center_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in [2, 3, 4]]
        center = df[center_cols].to_numpy()
        df["center_control_p1"] = (center == 1).sum(axis=1)
        df["center_control_p2"] = (center == 2).sum(axis=1)

        df["move_number"] = df["moveIndex"].astype(int)

        policy_cols = [f"policy_col_{i}" for i in range(7)]
        probs = df[policy_cols].to_numpy(dtype=float) + 1e-10
        probs /= probs.sum(axis=1, keepdims=True)

        df["policy_entropy"] = -(probs * np.log(probs)).sum(axis=1)
        df["policy_max"] = probs.max(axis=1)

        q_cols = [f"q_value_col_{i}" for i in range(7)]
        q = df[q_cols].to_numpy(dtype=float)

        df["qvalue_mean"] = q.mean(axis=1)
        df["qvalue_range"] = q.max(axis=1) - q.min(axis=1)

        return df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    def select_features(self, df: pd.DataFrame):
        board_cols = [f"board_before_r{r}c{c}" for r in range(6) for c in range(7)]
        policy_cols = [f"policy_col_{i}" for i in range(7)]
        q_cols = [f"q_value_col_{i}" for i in range(7)]

        engineered = [
            "player1_pieces", "player2_pieces",
            "center_control_p1", "center_control_p2",
            "move_number",
            "policy_entropy", "policy_max",
            "qvalue_mean", "qvalue_range",
        ]

        self.feature_columns = board_cols + policy_cols + q_cols + engineered

        X = df[self.feature_columns]
        y = self.build_outcome_target(df)
        return X, y, df["gameId"]

    # ------------------------------------------------------------------
    # Inference transform (CRITICAL)
    # ------------------------------------------------------------------
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("Preprocessor not fitted")

        df = self.clean_data(df)
        df = self.engineer_features(df)

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing inference features: {missing}")

        X = df[self.feature_columns]
        return self.scaler.transform(X)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
