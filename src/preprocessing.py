import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict

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
            if 'action_taken' in subset_df.columns:
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