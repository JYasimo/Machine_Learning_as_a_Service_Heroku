from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from joblib import load

router = APIRouter()

# Load the lgb model
lgb_model = load('../models/lgb_model.joblib')

# Define Pydantic models for prediction
class PredictionInput(BaseModel):
    store_id: str
    item_id: str
    day: int
    month: int
    year: int

class Prediction(BaseModel):
    date: str
    predicted_sales: float

# Define functions for prediction
# (preprocess_input_data and predict_sales)

@router.get("/sales/stores/items/")
def sales_prediction(input_data: PredictionInput):
    try:
        # Call the predict_sales function with the input data
        prediction = predict_sales(input_data.store_id, input_data.item_id, input_data.day, input_data.month, input_data.year)

        return prediction
    except HTTPException as e:
        raise e
