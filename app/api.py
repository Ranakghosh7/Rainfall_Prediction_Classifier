from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict_df
import pandas as pd

app = FastAPI(title='Rainfall Prediction API')

MODEL_PATH = 'models/rainfall_best_model.joblib'
model = load_model(MODEL_PATH)


class Observation(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    cloud_cover: float
    month: int
    day: int
    prev_rain_mm: float


@app.post("/predict")
def predict(obs: Observation):
    df = pd.DataFrame([{
        "temperature": obs.temperature,
        "humidity": obs.humidity,
        "pressure": obs.pressure,
        "wind_speed": obs.wind_speed,
        "cloud_cover": obs.cloud_cover,
        "month": str(obs.month),
        "day": str(obs.day),
        "prev_rain_mm": obs.prev_rain_mm
    }])

    out = predict_df(model, df)
    return out.to_dict(orient="records")[0]
