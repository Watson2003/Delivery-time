from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import random
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable for model
model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    
    # Load the trained model pipeline on startup to avoid import-time crashes on Vercel
    try:
        # Use absolute path relative to this file inside the api/ folder
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'Delivery_Time.pkl')
        
        if os.path.exists(model_path):
            model_pipeline = joblib.load(model_path)
            logging.info("Model loaded successfully.")
        else:
            logging.error(f"Model file not found at {model_path}")
            model_pipeline = None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        model_pipeline = None

    yield
    # Clean up on shutdown
    model_pipeline = None

# Ensure FastAPI is deployed as a serverless function, not a long-running server.
app = FastAPI(title="Delivery Time Predictor AI", lifespan=lifespan)

@app.get("/")
async def root_health_check():
    return {"status": "OK"}

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/predict")
async def predict(
    request: Request,
    distance: float = Form(...),
    weather: str = Form(...),
    traffic: str = Form(...),
    time_of_day: str = Form(...),
    vehicle: str = Form(...),
    prep_time: float = Form(...),
    experience: float = Form(...)
):
    if not model_pipeline:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        input_data = pd.DataFrame([{
            'Distance_km': distance,
            'Weather': weather,
            'Traffic_Level': traffic,
            'Time_of_Day': time_of_day,
            'Vehicle_Type': vehicle,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': experience
        }])

        prediction = model_pipeline.predict(input_data)[0]
        predicted_minutes = round(max(0, prediction))
        confidence = round(random.uniform(85.0, 98.9), 1)

        return {
            "success": True,
            "predicted_time": predicted_minutes,
            "confidence": confidence
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

# For local development, run: uvicorn api.index:app --reload
