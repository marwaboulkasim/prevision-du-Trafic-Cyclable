from fastapi import FastAPI
from predict import load_model, predict_traffic
from common.database.database import supabase
import datetime

app = FastAPI()

# Chargement du modèle
model = load_model()

@app.get("/predict")
def predict():
    predictions = predict_traffic(model)

    if not predictions:
        return {"error": "Pas de données disponibles pour la veille"}

    return {
        "message": "Prédictions générées et sauvegardées avec succès",
        "count": len(predictions),
        "predictions": predictions
    }


