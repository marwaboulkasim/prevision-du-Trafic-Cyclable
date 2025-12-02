from fastapi import APIRouter
from pred_client import get_prediction
from app import PredictionResponse

router = APIRouter()

@router.get("/prediction/{counter_id}", response_model=PredictionResponse)
def prediction_route(counter_id: str, date: str = None):
    """Appelle le servic ede prédiction et retourne la prédiction"""
    result = get_prediction(counter_id, date)

    if result is None:
        return {"error": "service de prediction indisponible"}
    
    return result
