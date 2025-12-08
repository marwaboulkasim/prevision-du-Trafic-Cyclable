
from fastapi import FastAPI
from predict import load_model, predict_traffic

app = FastAPI(
    title="API de prédiction trafic vélo - Journalier"
)

# Chargement du modèle
model = load_model()


@app.get("/predict")
def predict(counter_id: str, date: str):

    predictions = predict_traffic(model, date)

    # On récupère uniquement la prédiction du compteur demandé
    filter = [
        p for p in predictions if p["counter_id"] == counter_id
    ]

    if not filter:
        return {
            "counter_id": counter_id,
            "date": date,
            "error": "Aucune donnée pour ce compteur et cette date"
        }

    return {
        "counter_id": counter_id,
        "date": date,
        "prediction": filter[0]["forecast"]
    }
