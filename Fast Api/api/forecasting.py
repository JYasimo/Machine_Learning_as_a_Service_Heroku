from fastapi import APIRouter, Form, Request
from datetime import datetime, timedelta
from joblib import load
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="../templates")

# Load the SARIMA model
arima_model = load('../models/arima_model.joblib')

# Function to forecast sales for the next 7 days
def forecast_sales(input_date):
    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    forecast = arima_model.get_forecast(steps=7)
    forecast_values = forecast.predicted_mean.tolist()
    forecast_dict = {str(input_date + timedelta(days=i)): forecast_values[i] for i in range(7)}
    return forecast_dict

@router.get("/")
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/sales/national/")
async def national_sales_forecast(input_date: str = Form(...)):
    try:
        sales_forecast = forecast_sales(input_date)
        return {"sales_forecast": sales_forecast}
    except Exception as e:
        return {"error": str(e)}
