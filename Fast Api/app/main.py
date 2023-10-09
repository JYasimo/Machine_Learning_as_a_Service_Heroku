from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from datetime import datetime, timedelta
from joblib import load
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

# Create a Pydantic model for the response
class Prediction(BaseModel):
    date: str
    forecast: float

# Create a Pydantic model for the prediction input
class PredictionInput(BaseModel):
    input_date: str
    num_days: int = 7





@app.get("/", response_class=HTMLResponse)
def read_root():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Project Objectives</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }

            h1 {
                color: #333;
                text-align: center;
            }

            h2 {
                color: #666;
            }

            p {
                color: #444;
            }

            ul {
                list-style-type: circle;
                margin-left: 20px;
            }

            .endpoint {
                margin-top: 20px;
                margin-left: 40px;
            }

            a {
                color: #007BFF;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <h1>Project Objectives:</h1>
        <p>The project aims to develop and deploy two distinct machine learning models for an American retailer with 10 stores across 3 states, selling items from three categories. The objectives are:</p>
        <ol>
            <li>Build a predictive model: Develop a machine learning algorithm that accurately predicts sales revenue for a specific item in a particular store on a given date. This model will support inventory management and pricing decisions.</li>
            <li>Build a forecasting model: Create a time-series analysis algorithm to forecast the total sales revenue across all stores and items for the next 7 days. This model will provide insights for supply chain management and overall business planning.</li>
        </ol>

        <h2>List of Endpoints:</h2>

        <div class="endpoint">
            <h3>Predictive Model Endpoint:</h3>
            <ul>
                <li><strong>Endpoint:</strong> /sales/stores/items/</li>
                <li><strong>Description:</strong> Predicts the sales revenue for a given item in a specific store at a given date.</li>
                <li><strong>HTTP Method:</strong> POST</li>
                <li><strong>Input Parameters:</strong>
                    <ul>
                        <li>store_id: Identifier for the store (e.g., CA1, TX2).</li>
                        <li>item_id: Identifier for the item.</li>
                        <li>date: Date for which the sales prediction is required.</li>
                    </ul>
                </li>
                <li><strong>Output Format:</strong> JSON response containing the predicted sales revenue.</li>
                <li><strong>Performance Metrics:</strong> RMSE</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>Forecasting Model Endpoint:</h3>
            <ul>
                <li><strong>Endpoint:</strong> /sales/national/</li>
                <li><strong>Description:</strong> Forecasts the total sales revenue across all stores and items for the next 7 days.</li>
                <li><strong>HTTP Method:</strong> POST</li>
                <li><strong>Input Parameters:</strong>
                    <ul>
                        <li>start_date: The starting date for the forecasting period.</li>
                    </ul>
                </li>
                <li><strong>Output Format:</strong> JSON response containing the forecasted sales revenue for the next 7 days, broken down by date.</li>
                <li><strong>Performance Metrics:</strong> RMSE</li>
            </ul>
        </div>

        <h2>Link to Git repo:</h2>
        <p><a href="https://github.com/JYasimo/American_retailer" target="_blank">GitHub Repository</a></p>
    </body>
    </html>
    """
    return content



@app.get("/health/", response_class=HTMLResponse)
def health_check():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Check</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #007BFF; /* Blue background color */
                color: #fff; /* White text color */
            }

            h1 {
                color: #fff; /* White text color */
                text-align: center;
            }

            p {
                color: #fff; /* White text color */
                text-align: center;
            }
        </style>
    </head>
    <body>
        <h1>Welcome to the American Retailer API!</h1>
        <p>Have fun!</p>
    </body>
    </html>
    """
    return content

# Load the ARIMA model
arima_model = joblib.load('../models/arima_model.joblib')

# Function to forecast sales for the next 7 days
def forecast_sales(input_date):
    try:
        # Parse the input date string into a datetime object
        input_date = datetime.strptime(input_date, "%Y-%m-%d")

        # Make the ARIMA forecast for the next 7 days
        forecast = arima_model.get_forecast(steps=7)

        # Extract the forecasted values for the next 7 days
        forecast_values = forecast.predicted_mean.tolist()

        # Format the forecasted values as a dictionary
        forecast_dict = {str(input_date + timedelta(days=i)): forecast_values[i] for i in range(7)}

        return forecast_dict
    except Exception as e:
        raise Exception("Invalid date format or model error: " + str(e))

# Define the '/sales/forecast/' endpoint
@app.route('/sales/forecast/', methods=['POST'])
def sales_forecast():
    try:
        # Get the input date from the request
        input_date = request.form.get('date')
        
        # Call the forecast_sales function with the input date
        sales_forecast = forecast_sales(input_date)
        
        return {"sales_forecast": sales_forecast}
    except Exception as e:
        return {"error": str(e)}



# Load the lgb model
lgb_model = load('../models/lgb_model.joblib')

from pydantic import BaseModel

# Create a Pydantic model for the response
class Prediction(BaseModel):
    date: str
    predicted_sales: float

# Create a Pydantic model for the prediction input
class PredictionInput(BaseModel):
    store_id: str
    item_id: str
    day: int
    month: int
    year: int

# Preprocess input data to match model requirements
def preprocess_input_data(store_id, item_id, day, month, year):
    try:
        # Combine day, month, and year inputs into a date string
        input_date = f"{year}-{month:02d}-{day:02d}"

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({'store_id': [store_id], 'item_id': [item_id], 'date': [input_date]})

        # Ensure item_id and store_id are treated as categorical columns
        input_data['item_id'] = input_data['item_id'].astype('category')
        input_data['store_id'] = input_data['store_id'].astype('category')

        return input_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Predict sales based on preprocessed input data
def predict_sales(store_id, item_id, day, month, year):
    try:
        # Preprocess the input data
        preprocessed_data = preprocess_input_data(store_id, item_id, day, month, year)

        # Use the loaded lgb model to make the prediction
        prediction = lgb_model.predict(preprocessed_data)

        # Format the prediction as a dictionary
        result = {"date": preprocessed_data['date'][0], "predicted_sales": float(prediction[0])}

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Define the '/sales/stores/items/' endpoint
@app.get("/sales/stores/items/")
def sales_prediction(input_data: PredictionInput):
    try:
        # Call the predict_sales function with the input data
        prediction = predict_sales(input_data.store_id, input_data.item_id, input_data.day, input_data.month, input_data.year)

        return prediction
    except HTTPException as e:
        raise e
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", lifespan="on")


from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from datetime import datetime, timedelta
from joblib import load
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

# Create a Pydantic model for the response
class Prediction(BaseModel):
    date: str
    forecast: float

# Create a Pydantic model for the prediction input
class PredictionInput(BaseModel):
    input_date: str
    num_days: int = 7





@app.get("/", response_class=HTMLResponse)
def read_root():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Project Objectives</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }

            h1 {
                color: #333;
                text-align: center;
            }

            h2 {
                color: #666;
            }

            p {
                color: #444;
            }

            ul {
                list-style-type: circle;
                margin-left: 20px;
            }

            .endpoint {
                margin-top: 20px;
                margin-left: 40px;
            }

            a {
                color: #007BFF;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <h1>Project Objectives:</h1>
        <p>The project aims to develop and deploy two distinct machine learning models for an American retailer with 10 stores across 3 states, selling items from three categories. The objectives are:</p>
        <ol>
            <li>Build a predictive model: Develop a machine learning algorithm that accurately predicts sales revenue for a specific item in a particular store on a given date. This model will support inventory management and pricing decisions.</li>
            <li>Build a forecasting model: Create a time-series analysis algorithm to forecast the total sales revenue across all stores and items for the next 7 days. This model will provide insights for supply chain management and overall business planning.</li>
        </ol>

        <h2>List of Endpoints:</h2>

        <div class="endpoint">
            <h3>Predictive Model Endpoint:</h3>
            <ul>
                <li><strong>Endpoint:</strong> /sales/stores/items/</li>
                <li><strong>Description:</strong> Predicts the sales revenue for a given item in a specific store at a given date.</li>
                <li><strong>HTTP Method:</strong> POST</li>
                <li><strong>Input Parameters:</strong>
                    <ul>
                        <li>store_id: Identifier for the store (e.g., CA1, TX2).</li>
                        <li>item_id: Identifier for the item.</li>
                        <li>date: Date for which the sales prediction is required.</li>
                    </ul>
                </li>
                <li><strong>Output Format:</strong> JSON response containing the predicted sales revenue.</li>
                <li><strong>Performance Metrics:</strong> RMSE</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>Forecasting Model Endpoint:</h3>
            <ul>
                <li><strong>Endpoint:</strong> /sales/national/</li>
                <li><strong>Description:</strong> Forecasts the total sales revenue across all stores and items for the next 7 days.</li>
                <li><strong>HTTP Method:</strong> POST</li>
                <li><strong>Input Parameters:</strong>
                    <ul>
                        <li>start_date: The starting date for the forecasting period.</li>
                    </ul>
                </li>
                <li><strong>Output Format:</strong> JSON response containing the forecasted sales revenue for the next 7 days, broken down by date.</li>
                <li><strong>Performance Metrics:</strong> RMSE</li>
            </ul>
        </div>

        <h2>Link to Git repo:</h2>
        <p><a href="https://github.com/JYasimo/American_retailer" target="_blank">GitHub Repository</a></p>
    </body>
    </html>
    """
    return content



@app.get("/health/", response_class=HTMLResponse)
def health_check():
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Check</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #007BFF; /* Blue background color */
                color: #fff; /* White text color */
            }

            h1 {
                color: #fff; /* White text color */
                text-align: center;
            }

            p {
                color: #fff; /* White text color */
                text-align: center;
            }
        </style>
    </head>
    <body>
        <h1>Welcome to the American Retailer API!</h1>
        <p>Have fun!</p>
    </body>
    </html>
    """
    return content

# Load the ARIMA model
arima_model = joblib.load('../models/arima_model.joblib')

# Function to forecast sales for the next 7 days
def forecast_sales(input_date):
    try:
        # Parse the input date string into a datetime object
        input_date = datetime.strptime(input_date, "%Y-%m-%d")

        # Make the ARIMA forecast for the next 7 days
        forecast = arima_model.get_forecast(steps=7)

        # Extract the forecasted values for the next 7 days
        forecast_values = forecast.predicted_mean.tolist()

        # Format the forecasted values as a dictionary
        forecast_dict = {str(input_date + timedelta(days=i)): forecast_values[i] for i in range(7)}

        return forecast_dict
    except Exception as e:
        raise Exception("Invalid date format or model error: " + str(e))

# Define the '/sales/forecast/' endpoint
@app.route('/sales/forecast/', methods=['POST'])
def sales_forecast():
    try:
        # Get the input date from the request
        input_date = request.form.get('date')
        
        # Call the forecast_sales function with the input date
        sales_forecast = forecast_sales(input_date)
        
        return {"sales_forecast": sales_forecast}
    except Exception as e:
        return {"error": str(e)}



# Load the lgb model
lgb_model = load('../models/lgb_model.joblib')

from pydantic import BaseModel

# Create a Pydantic model for the response
class Prediction(BaseModel):
    date: str
    predicted_sales: float

# Create a Pydantic model for the prediction input
class PredictionInput(BaseModel):
    store_id: str
    item_id: str
    day: int
    month: int
    year: int

# Preprocess input data to match model requirements
def preprocess_input_data(store_id, item_id, day, month, year):
    try:
        # Combine day, month, and year inputs into a date string
        input_date = f"{year}-{month:02d}-{day:02d}"

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({'store_id': [store_id], 'item_id': [item_id], 'date': [input_date]})

        # Ensure item_id and store_id are treated as categorical columns
        input_data['item_id'] = input_data['item_id'].astype('category')
        input_data['store_id'] = input_data['store_id'].astype('category')

        return input_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Predict sales based on preprocessed input data
def predict_sales(store_id, item_id, day, month, year):
    try:
        # Preprocess the input data
        preprocessed_data = preprocess_input_data(store_id, item_id, day, month, year)

        # Use the loaded lgb model to make the prediction
        prediction = lgb_model.predict(preprocessed_data)

        # Format the prediction as a dictionary
        result = {"date": preprocessed_data['date'][0], "predicted_sales": float(prediction[0])}

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Define the '/sales/stores/items/' endpoint
@app.get("/sales/stores/items/")
def sales_prediction(input_data: PredictionInput):
    try:
        # Call the predict_sales function with the input data
        prediction = predict_sales(input_data.store_id, input_data.item_id, input_data.day, input_data.month, input_data.year)

        return prediction
    except HTTPException as e:
        raise e
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", lifespan="on")


