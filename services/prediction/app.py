from fastapi import FastAPI
from predict import predict_for_day

app = FastAPI(
    title = "API de prédiction trafic vélo - Journalier"
)

@app.get("/predict")
def predict(counter_id: str, date: str = None):
    """Retourne la prediction de trafic pour une journée"""
    return predict_for_day
