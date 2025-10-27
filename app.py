from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import catboost
import xgboost
import joblib

app = FastAPI(title="Flight Disruption Prediction API")

# Mock models (in actual use, you'd load pre-trained models)
# model_cancel = catboost.CatBoostClassifier().load_model("catboost_cancel.cbm")
# model_delay = xgboost.XGBRegressor()
# model_delay.load_model("xgb_delay.json")

# For demonstration, fake predictions
class FlightInput(BaseModel):
    origin: str
    dest: str
    airline: str
    month: int
    day_of_week: int
    sched_dep_hour: int
    distance: float
    temperature_origin: float
    wind_speed_origin: float
    visibility_origin: float
    temperature_dest: float
    wind_speed_dest: float
    visibility_dest: float

@app.post("/predict")
def predict_flight_disruption(data: FlightInput):
    # Normally you’d preprocess and feed to model
    # For demo: random but deterministic-looking results
    np.random.seed(sum(map(ord, data.origin + data.dest)) % 100)
    cancel_prob = np.random.uniform(0.05, 0.95)
    delay_minutes = np.random.uniform(0, 90)

    return {
        "cancellation_probability": round(float(cancel_prob), 3),
        "expected_delay_minutes": round(float(delay_minutes), 2)
    }
