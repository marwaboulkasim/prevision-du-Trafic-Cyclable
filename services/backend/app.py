from fastapi import FastAPI
from pred_client import get_prediction


app = FastAPI()

@app.get("/traffic/predict")
def predict(counter_id: str, date:str):
    """Le endpoint utiliser par le frontend"""

    result = get_prediction(counter_id, date)

    if "error" in result:
        return {
            "status": "error",
            "message": "service de prediction indisponible"
        }
    
    return {
        "status": "success",
        "counter_id": counter_id,
        "date": date,
        "prediction": result["prediction"]
    }




