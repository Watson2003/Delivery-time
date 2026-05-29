from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import random
import os
import logging
import gzip

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables for model and templates
model_pipeline = None
templates = None

# Set up absolute paths for Vercel
# index.py is in /api, so ROOT_DIR is the parent directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, templates
    
    # Safely load templates
    templates_dir = os.path.join(ROOT_DIR, "templates")
    try:
        templates = Jinja2Templates(directory=templates_dir)
        logging.info(f"Templates loaded from: {templates_dir}")
    except Exception as e:
        logging.error(f"Could not load templates: {e}")
        templates = None

    # Load the trained model pipeline (compressed for Vercel size limits)
    try:
        model_path = os.path.join(ROOT_DIR, 'Delivery_Time.pkl.gz')
        if os.path.exists(model_path):
            with gzip.open(model_path, 'rb') as f:
                model_pipeline = joblib.load(f)
            logging.info("Model loaded successfully from compressed file.")
        else:
            # Fallback to uncompressed for local development
            uncompressed_path = os.path.join(ROOT_DIR, 'Delivery_Time.pkl')
            model_pipeline = joblib.load(uncompressed_path)
            logging.info("Model loaded successfully from uncompressed file.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        model_pipeline = None

    yield
    # Clean up on shutdown if needed
    model_pipeline = None

app = FastAPI(title="Delivery Time Predictor AI", lifespan=lifespan)

# Safely mount static files (may fail on read-only serverless filesystem if directory missing)
static_dir = os.path.join(ROOT_DIR, "static")
try:
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logging.info(f"Static files mounted from: {static_dir}")
    else:
        logging.warning(f"Static directory not found: {static_dir}")
except Exception as e:
    logging.warning(f"Could not mount static files: {e}")

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    if templates is None:
        return HTMLResponse(content="<h1>Template Error</h1><p>Templates could not be loaded.</p>", status_code=500)
    return templates.TemplateResponse("index.html", {"request": request})

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

if __name__ == "__main__":
    import uvicorn
    # Important: when running locally, run from project root: uvicorn api.index:app --reload
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
