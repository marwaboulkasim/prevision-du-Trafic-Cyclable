import joblib
import datetime
import numpy as np

# Charger le modéle sauvegardé
model = joblib.load("model.pkl")

def predict_for_day(counter_id: str, date: str = None):
    
    if date is None:
        date = datetime.date.today().isoformat()

    date_objet = datetime.datetime.fromisoformat(date)

    month = date_objet.month
    weekday = date_objet.weekday()

    X = np.array([[month, weekday]])
    prediction = model.predict(X)[0]

    return {
        "counter_id": counter_id,
        "date": date,
        "prediction": float(prediction)
    }