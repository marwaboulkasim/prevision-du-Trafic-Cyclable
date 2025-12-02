from fastapi import FastAPI
from routes import router
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    counter_id: str
    date: str
    prediction: float

app = FastAPI(title="Backend API - Traffic Vélo")

app.include_router(router)


