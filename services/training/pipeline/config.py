from pathlib import Path


# Chemins

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/raw"
MODEL_DIR = BASE_DIR.parent / "models"
OUTPUT_DIR = BASE_DIR.parent / "outputs"





TARGET_COLUMN = "value"        
DATE_COLUMN = "time"           
CATEGORICAL_COLUMNS = ["id"]   # seule colonne catégorielle existante
NUMERIC_COLUMNS = ["value"]
FEATURE_COLUMNS = ["id"]       # tu peux ajouter "coordinates" si besoin
DATE_FEATURES = ["year", "month", "day", "hour", "weekday"]

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2




# Paramètres entraînement

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


# Paramètres XGBoost

MODEL_TYPE = "XGBoost"
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED
}


# Préprocessing

DATE_FEATURES = ["year", "month", "day", "hour", "weekday"]
CATEGORICAL_COLUMNS = ["vehicleType", "laneId"]
NUMERIC_COLUMNS = ["count"]


# Hyperparam tuning

HYPERPARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}
