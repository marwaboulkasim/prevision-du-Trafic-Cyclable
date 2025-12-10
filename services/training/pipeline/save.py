import joblib
import json
from pathlib import Path

def save_model(model, path="models/xgb_model.pkl"):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")

def save_encoder(encoder, path="models/encoder.pkl"):
    joblib.dump(encoder, path)
    print(f"Encoder sauvegardé dans {path}")

def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques sauvegardées dans {path}")
