import requests

PREDICTION_URL = "http://localhost:8000/predict"

def get_prediction(counter_id: str, date: str = None):
    params= {
        "counter_id": counter_id,
        "date": date
    }
    try:
        response = requests.get(PREDICTION_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return{
            "error": "Service prediction inaccessible",
            "details": str(e)
        }
    