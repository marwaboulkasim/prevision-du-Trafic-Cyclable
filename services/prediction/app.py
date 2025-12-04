from fastapi import FastAPI
from predict import load_model, predict_traffic
from common.database.database import supabase

app = FastAPI(
    title = "API de prédiction trafic vélo - Journalier"
)

# Chargement de modéle
model = load_model()

@app.get("/predict")
def predict():
    """Retourne la prediction de trafic pour une journée"""

    prediction = predict_traffic(model)
    # response = (
    #     supabase.table("forecast_data").update({"forecast":"prediction"}).eq
    # )

    return {
        "prediction": prediction
    }

