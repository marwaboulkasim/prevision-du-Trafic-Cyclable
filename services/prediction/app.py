from fastapi import FastAPI
from predict import load_model, predict_traffic
from common.database.database import supabase
import pandas as pd 
from typing import Optional

app = FastAPI(
    title="API de prédiction trafic vélo - Journalier"
)

# Chargement du modèle au démarrage
model = load_model()

@app.get("/predict")
def predict(
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    hour: Optional[int] = None,
    weekday: Optional[int] = None,
    counter_id: str = None
):
    """
    Retourne la prédiction du trafic cyclable.
    
    - Si aucun paramètre n’est fourni → prédictions journalières (ton ancien comportement)
    - Si des paramètres sont fournis → prédiction unitaire (pour Streamlit)
    """

    # 🔹 Cas 1 : prédiction journalière (ton comportement actuel)
    if all(v is None for v in [year, month, day, hour, weekday]):
        prediction = predict_traffic(model)
        return {"prediction": prediction}

    # 🔹 Cas 2 : prédiction *unitaire* pour Streamlit
    features = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "weekday": weekday
    }

    # Format attendu par le modèle
    df = pd.DataFrame([features])

    forecast = float(model.predict(df)[0])

    return {
        "prediction": [
            {
                "counter_id": counter_id,
                "forecast": forecast
            }
        ]
    }
