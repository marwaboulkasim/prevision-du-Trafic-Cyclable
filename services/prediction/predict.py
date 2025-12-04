import pandas as pd
import pickle
from common.database.database import supabase
import datetime


def load_model():
    """charge le modéle entrainé"""
    model_path = "models/xgb_model.pkl"
    with open(model_path, "rb") as f:
       model = pickle.load(f)
    return model


def fetch_daily_data():
    today = datetime.date.today().isoformat()
    daily_data = supabase.table("forecast_data") \
        .select("*") \
        .gte("datetime", f"{today}T00:00:00") \
        .lt("datetime", f"{today}T23:59:59") \
        .execute()
    return daily_data

def predict_traffic(model):
    """Génére la prediction du traffic"""
    daily_data = fetch_daily_data()
    print(daily_data)

    df = pd.json_normalize(daily_data.data)
    print(f"la base de donnée", df)
    
    df = df[["hour", "year",  "weekday", "day", "month", "counter_id"]]

    def predict_row(row):
        features = row.drop("counter_id")            
        features = features.to_frame().T             
        pred = model.predict(features)[0]            
        return float(pred) 

    df["forecast"]= df.apply(predict_row, axis=1)
    print(df)
    # prediction = model.predict(df.drop(column= "counter_id"))
    # print(prediction)
    # return prediction.tolist()

print(predict_traffic(load_model()))





 # # Convertir les colonnes en types numériques (sinon XGBoost refuse les 'object')
    # numeric_cols = ["hour", "year", "weekday", "day", "month"]
    # for col in numeric_cols:
    #     df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # # Vérifier s'il y a des NaN après la conversion
    # if df[numeric_cols].isnull().any().any():
    #     print("⚠️ Attention: Certaines valeurs n'ont pas pu être converties en nombres")
    #     df = df.dropna(subset=numeric_cols)