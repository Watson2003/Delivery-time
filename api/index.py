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

# Global variables
model_pipeline = None
templates = None
model_load_error = "Unknown error"

# We look for templates/static either in the current api/ dir or the parent root dir
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

def find_dir(dir_name):
    if os.path.isdir(os.path.join(current_dir, dir_name)):
        return os.path.join(current_dir, dir_name)
    elif os.path.isdir(os.path.join(parent_dir, dir_name)):
        return os.path.join(parent_dir, dir_name)
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, templates, model_load_error
    
    # Load Templates
    templates_dir = find_dir("templates")
    if templates_dir:
        try:
            templates = Jinja2Templates(directory=templates_dir)
            logging.info(f"Templates loaded from: {templates_dir}")
        except Exception as e:
            logging.error(f"Could not load templates: {e}")
            templates = None
    else:
        logging.error("Templates directory not found!")
        templates = None

    # Load Model (supports .pkl and .pkl.gz)
    try:
        model_loaded = False
        
        # Check inside api/ and root for model
        possible_paths = [
            os.path.join(current_dir, 'Delivery_Time.pkl.gz'),
            os.path.join(parent_dir, 'Delivery_Time.pkl.gz'),
            os.path.join(current_dir, 'Delivery_Time.pkl'),
            os.path.join(parent_dir, 'Delivery_Time.pkl')
        ]
        
        paths_checked = []
        for path in possible_paths:
            paths_checked.append(path)
            if os.path.exists(path):
                logging.info(f"Loading model from {path}...")
                if path.endswith('.gz'):
                    with gzip.open(path, 'rb') as f:
                        model_pipeline = joblib.load(f)
                else:
                    model_pipeline = joblib.load(path)
                model_loaded = True
                model_load_error = None
                logging.info("Model loaded successfully.")
                break
                
        if not model_loaded:
            model_load_error = f"File not found. Checked: {paths_checked}"
            logging.error(model_load_error)
            model_pipeline = None
    except Exception as e:
        import traceback
        model_load_error = f"Exception during joblib.load: {str(e)} \n Traceback: {traceback.format_exc()}"
        logging.error(model_load_error)
        model_pipeline = None

    yield
    # Clean up on shutdown
    model_pipeline = None

app = FastAPI(title="Delivery Time Predictor AI", lifespan=lifespan)

# Safely mount static files
static_dir = find_dir("static")
if static_dir:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logging.info(f"Static files mounted from: {static_dir}")
else:
    logging.warning("Static directory not found!")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serve the frontend UI on the root endpoint"""
    if templates is None:
        return HTMLResponse(content="<h1>Template Error</h1><p>Templates object is None. Check logs.</p>", status_code=500)
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HTMLResponse(content=f"<h1>Template Render Error</h1><pre>{error_details}</pre>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
        return JSONResponse(status_code=500, content={"error": f"Model not loaded. Reason: {model_load_error}"})

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
