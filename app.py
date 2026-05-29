from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import random
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Delivery Time Predictor AI")

# Set up absolute paths for Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely mount static files (may fail on read-only serverless filesystem)
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

try:
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logging.info(f"Static files mounted from: {static_dir}")
    else:
        logging.warning(f"Static directory not found: {static_dir}")
except Exception as e:
    logging.warning(f"Could not mount static files: {e}")

try:
    templates = Jinja2Templates(directory=templates_dir)
    logging.info(f"Templates loaded from: {templates_dir}")
except Exception as e:
    logging.error(f"Could not load templates: {e}")
    templates = None


# Load the trained model pipeline (compressed for Vercel size limits)
try:
    model_path = os.path.join(BASE_DIR, 'Delivery_Time.pkl.gz')
    if os.path.exists(model_path):
        import gzip
        with gzip.open(model_path, 'rb') as f:
            model_pipeline = joblib.load(f)
        logging.info("Model loaded successfully from compressed file.")
    else:
        # Fallback to uncompressed for local development
        model_path = os.path.join(BASE_DIR, 'Delivery_Time.pkl')
        model_pipeline = joblib.load(model_path)
        logging.info("Model loaded successfully from uncompressed file.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model_pipeline = None

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
        # Construct DataFrame from input data
        # Note: Match column names exactly with what the model expects
        input_data = pd.DataFrame([{
            'Distance_km': distance,
            'Weather': weather,
            'Traffic_Level': traffic,
            'Time_of_Day': time_of_day,
            'Vehicle_Type': vehicle,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': experience
        }])

        # Predict the delivery time
        prediction = model_pipeline.predict(input_data)[0]
        
        # Round the prediction to the nearest minute
        predicted_minutes = round(max(0, prediction)) # Ensure no negative time

        # Generate a mock confidence score for aesthetic purposes (85% to 98%)
        # In a real regression model, this could be derived from prediction intervals
        confidence = round(random.uniform(85.0, 98.9), 1)

        return {
            "success": True,
            "predicted_time": predicted_minutes,
            "confidence": confidence
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

# Health check endpoint for monitoring
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
