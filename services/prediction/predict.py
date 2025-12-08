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


def fetch_daily_data(date: str):
    """
    Récupère les données depuis Supabase pour une date choisie
    Format attendu : YYYY-MM-DD
    """
    start = f"{date}T00:00:00"
    end = f"{date}T23:59:59"

    daily_data = (
        supabase.table("forecast_data")
        .select("*")
        .gte("datetime", start)
        .lt("datetime", end)
        .execute()
    )
    return daily_data


def predict_traffic(model, date: str):
    """Génère la prédiction du trafic vélo pour la date choisie"""
    
    daily_data = fetch_daily_data(date)

    if not daily_data.data:
        return []

    df = pd.json_normalize(daily_data.data)

    cols = ["hour", "year", "weekday", "day", "month", "counter_id"]
    df = df[cols]

    # Conversion en int propre
    for col in ["hour", "year", "weekday", "day", "month"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[["hour", "year", "weekday", "day", "month"]] = (df[["hour", "year", "weekday", "day", "month"]].fillna(0).astype(int))

    def predict_row(row):
        features = row[["year", "month", "day", "hour", "weekday"]].to_frame().T
        features = features.astype(int)
        pred = model.predict(features)[0]
        return float(pred)

    df["forecast"] = df.apply(predict_row, axis=1)

    # Résultat propre pour l’API
    return df[["counter_id", "forecast"]].to_dict(orient="records")


# Test local
if __name__ == "__main__":
    model = load_model()
    predictions = predict_traffic(model, "2025-11-22")
    print("Prédictions :", predictions)










