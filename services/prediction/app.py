from fastapi import FastAPI, Query
from predict import load_model, predict_traffic

app = FastAPI(
    title="API de prédiction trafic vélo - Journalier"
)

# Chargement du modèle une seule fois au démarrage
model = load_model()

@app.get("/predict")
def predict(
    year: int | None = Query(None),
    month: int | None = Query(None),
    day: int | None = Query(None),
    hour: int | None = Query(None),
    weekday: int | None = Query(None)
):
    """
    Retourne la prédiction de trafic.
    Si aucun paramètre n'est fourni, prédit pour les données du jour.
    Sinon, prédit pour les valeurs fournies.
    """
    prediction = predict_traffic(model, year, month, day, hour, weekday)
    return {"prediction": prediction}
