import pandas as pd
import pickle
from common.database.database import supabase
import datetime

def load_model():
    """Charge le modèle entraîné"""
    model_path = "models/xgb_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def fetch_daily_data():
    """Récupère les données du jour depuis Supabase"""
    today = datetime.date.today().isoformat()
    daily_data = supabase.table("forecast_data") \
        .select("*") \
        .gte("datetime", f"{today}T00:00:00") \
        .lt("datetime", f"{today}T23:59:59") \
        .execute()
    return daily_data

def predict_traffic(model, year=None, month=None, day=None, hour=None, weekday=None):
    """
    Génère la prédiction du trafic vélo.
    Si aucun paramètre n'est fourni, on utilise les données du jour.
    Sinon, on prédit pour l'entrée fournie (year, month, day, hour, weekday).
    """
    if all(v is not None for v in [year, month, day, hour, weekday]):
        # Création d'un DataFrame pour l'entrée personnalisée
        df = pd.DataFrame([{
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "hour": int(hour),
            "weekday": int(weekday),
            "counter_id": "input"
        }])
    else:
        # Prédiction sur les données du jour
        daily_data = fetch_daily_data()
        if not daily_data.data:
            return []
        df = pd.json_normalize(daily_data.data)
        cols = ["hour", "year", "weekday", "day", "month", "counter_id"]
        df = df[cols]
        for col in ["hour", "year", "weekday", "day", "month"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')  
        df[["hour", "year", "weekday", "day", "month"]] = df[["hour", "year", "weekday", "day", "month"]].fillna(0).astype(int)

    def predict_row(row):
        features = row[["year", "month", "day", "hour", "weekday"]].to_frame().T
        features = features.astype(int)  
        pred = model.predict(features)[0]
        return float(pred)

    df["forecast"] = df.apply(predict_row, axis=1)

    return df[["counter_id", "forecast"]].to_dict(orient="records")


if __name__ == "__main__":
    model = load_model()
    predictions = predict_traffic(model)
    print("Prédictions du jour:", predictions)
