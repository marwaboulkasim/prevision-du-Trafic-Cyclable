import pandas as pd
import pickle


def load_model():
    """charge le modéle entrainé"""
    model_path = "model/model.pkl"
    with open(model_path, "rb") as f:
       model = pickle.load(f)
    return model


def features(counter_id: str, date: str):

    df = pd.dataframe([{
        "counter_id": counter_id,
        "date": date,
        "day": pd.to_datetime(date).day,
        "month": pd.to_datetime(date).month,
        "weekday": pd.to_datetime(date).weekday(),

    }])
    return df

def predict_traffic(model, counter_id: str , date: str):
    """Génére la prediction du traffic"""
    features = features(counter_id, date)
    prediction = model.predict(features)[0]
    return float(prediction)











# def predict_for_day(counter_id: str, date: str = None):
    
#     if date is None:
#         date = datetime.date.today().isoformat()

#     date_objet = datetime.datetime.fromisoformat(date)

#     month = date_objet.month
#     weekday = date_objet.weekday()

#     X = np.array([[month, weekday]])
#     prediction = model.predict(X)[0]

#     return {
#         "counter_id": counter_id,
#         "date": date,
#         "prediction": float(prediction)
#     }