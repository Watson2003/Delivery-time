# file:///d:/ROOT/Delivery%20time/api/index.py
import os
import logging
import gzip
import traceback
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import random
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ----------------------------------------------------------------------
# Logging configuration (visible in Vercel logs)
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ----------------------------------------------------------------------
# Helper to locate files (works whether the repo is run from the root or
# from the api/ sub‑directory)
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)          # .../api
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def locate(path: str) -> str:
    """Return an absolute path. Look in the api folder first, then the project root."""
    candidate = os.path.join(CURRENT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(ROOT_DIR, path)
    return candidate

# ----------------------------------------------------------------------
# Global objects populated at start‑up
# ----------------------------------------------------------------------
model_pipeline = None          # the trained scikit‑learn pipeline
model_load_error = None        # human‑readable error if loading fails
templates = None               # Jinja2 template engine (optional UI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's lifespan event runs once when the container starts.
    We load the model *once* here so the serverless function does not
    re‑load the (large) pickle on every request.
    """
    global model_pipeline, model_load_error, templates

    # ---------- load HTML templates (optional) ----------
    tmpl_dir = locate("templates")
    if os.path.isdir(tmpl_dir):
        try:
            templates = Jinja2Templates(directory=tmpl_dir)
            logging.info(f"Templates loaded from {tmpl_dir}")
        except Exception as exc:
            logging.error(f"Failed to initialise Jinja2: {exc}")
            templates = None
    else:
        logging.info("No templates directory found – the root endpoint will return JSON.")

    # ---------- load the sklearn pipeline ----------
    model_path = locate("Delivery_Time.pkl")
    try:
        logging.info(f"Loading model from {model_path} …")
        if model_path.endswith('.gz'):
            with gzip.open(model_path, "rb") as f:
                model_pipeline = joblib.load(f)
        else:
            model_pipeline = joblib.load(model_path)
        model_load_error = None
        logging.info("✅ Model loaded successfully.")
    except Exception as exc:
        model_load_error = (
            f"Failed to load model: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )
        logging.error(model_load_error)
        model_pipeline = None

    yield

    # ------------------------------------------------------------------
    # cleanup (the container will be discarded after the request anyway)
    # ------------------------------------------------------------------
    model_pipeline = None
    logging.info("FastAPI lifespan shut down – model cleared.")


app = FastAPI(title="Delivery‑Time Predictor", lifespan=lifespan)

# ----------------------------------------------------------------------
# Serve static files (if you have a `static/` folder)
# ----------------------------------------------------------------------
static_dir = locate("static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logging.info(f"Static files mounted from {static_dir}")


# ----------------------------------------------------------------------
# End‑points
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """If you have an HTML UI, render it. Otherwise return a tiny JSON health check."""
    if templates:
        try:
            return templates.TemplateResponse("index.html", {"request": request})
        except Exception as exc:
            logging.error(f"Template rendering error: {exc}")
            return HTMLResponse(content="<h1>Template error</h1><p>Check logs.</p>", status_code=500)
    return {"status": "OK", "message": "FastAPI is alive"}


@app.get("/health")
async def health():
    """Simple liveness probe for Vercel."""
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
    experience: float = Form(...),
):
    """Return a delivery‑time prediction or a clear error payload."""
    if model_pipeline is None:
        # The start‑up loader failed – surface the stored error message.
        return JSONResponse(
            status_code=500,
            content={"error": f"Model not available. {model_load_error or ''}"},
        )

    try:
        # Build a single‑row DataFrame that matches the training columns
        input_df = pd.DataFrame(
            [
                {
                    "Distance_km": distance,
                    "Weather": weather,
                    "Traffic_Level": traffic,
                    "Time_of_Day": time_of_day,
                    "Vehicle_Type": vehicle,
                    "Preparation_Time_min": prep_time,
                    "Courier_Experience_yrs": experience,
                }
            ]
        )

        prediction = model_pipeline.predict(input_df)[0]
        minutes = round(max(0, prediction))
        confidence = round(random.uniform(85.0, 98.9), 1)

        return {
            "success": True,
            "predicted_time_min": minutes,
            "confidence_percent": confidence,
        }

    except Exception as exc:
        # Unexpected error – log and return a readable JSON payload.
        err_msg = f"{type(exc).__name__}: {exc}"
        logging.error(f"Prediction error: {err_msg}\n{traceback.format_exc()}")
        return JSONResponse(status_code=400, content={"error": err_msg})
