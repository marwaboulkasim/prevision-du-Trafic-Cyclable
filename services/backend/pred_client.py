import requests

PREDICTION_URL = "http://localhost:8000/predict"

def get_prediction(counter_id: str, date: str = None):
    params= {}
    if date:
        params["date"] = date
    
    url = f"{PREDICTION_URL}/{counter_id}"
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None
    return response.json()