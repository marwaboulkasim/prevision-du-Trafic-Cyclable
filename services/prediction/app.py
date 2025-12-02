from fastapi import FastAPI
from predict import load_model, predict_traffic

app = FastAPI(
    title = "API de prédiction trafic vélo - Journalier"
)

# Chargement de modéle
model = load_model()

@app.get("/predict")
def predict(counter_id: str, date: str = None):
    """Retourne la prediction de trafic pour une journée"""
    
    prediction = predict_traffic(model, counter_id, date)
    return {
        "counter_id": counter_id,
        "date": date,
        "prediction": prediction
    }
