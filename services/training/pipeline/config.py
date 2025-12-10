from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/raw"
MODEL_DIR = BASE_DIR.parent / "models"
OUTPUT_DIR = BASE_DIR.parent / "outputs"

# Colonnes et features
TARGET_COLUMN = "intensity"  
DATE_COLUMN = "date"   
ID_COLUMNS = "counter_id"

# Colonnes catégorielles
CATEGORICAL_COLUMNS = ["counter_id"]

# Colonnes numériques (features)
NUMERIC_COLUMNS = [
    "rolling_7d", "rolling_28d",
    "lag_7d", "lag_28d",
    "temperature", "rain",
    "hour", "day", "month", "year", "weekday",
    "is_weekend",
]

# Features pour le modèle
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

# Features issues de la date
DATE_FEATURES = ["year", "month", "day", "hour", "weekday"]

# Split & seed
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# XGBoost
MODEL_TYPE = "XGBoost"
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED
}

# Hyperparam tuning
HYPERPARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}
