import os
import joblib # Using joblib as we compressed with it
import numpy as np
import pandas # Although not directly used, good practice if model expects DataFrame features internally
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, conlist, validator
from typing import List, Tuple
import uvicorn
import sys # For stderr printing during debugging if needed

# --- Define Input/Output Data Structures ---

# Define a single station's input data structure for clarity and validation
class StationInput(BaseModel):
    pWaveTime: float = Field(..., description="Unix timestamp (can include milliseconds) of P-wave arrival")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Station latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Station longitude (-180 to 180)")

# Define the overall input as a list of stations, ensuring at least 3
StationListInput = conlist(StationInput, min_length=3)


# Define the expected output structure
class PredictionOutput(BaseModel):
    latitude: float = Field(..., description="Predicted epicenter latitude")
    longitude: float = Field(..., description="Predicted epicenter longitude")

# --- Model Loading ---
MODEL_FILENAME = "XGBoost_model.pkl" # Use the compressed name
model = None
app = FastAPI(title="Earthquake Epicenter Prediction API") # Create FastAPI app

@app.on_event("startup")
def load_model():
    """Load the compressed model at startup."""
    global model
    # Construct path relative to this script file
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    print(f"INFO: Attempting to load model from: {model_path}", file=sys.stderr)

    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found at {model_path}", file=sys.stderr)
        # Don't assign model, endpoint checks will fail later
        return

    try:
        model = joblib.load(model_path)
        # Optionally check the loaded model type or attributes here
        print(f"INFO: Model loaded successfully. Type: {type(model)}", file=sys.stderr)
        # Example check (adapt if needed):
        # if not hasattr(model, 'predict'):
        #     print("FATAL ERROR: Loaded object does not have a 'predict' method.", file=sys.stderr)
        #     model = None # Ensure model is None if invalid
    except Exception as e:
        print(f"FATAL ERROR: Error loading model from {model_path}: {e}", file=sys.stderr)
        # Ensure model is None on error
        model = None


@app.get("/", summary="Health Check")
def read_root():
    """Basic health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


# Use Body(..., embed=True) if you expect {"stations": [[...], ...]}
# Use just StationListInput if you expect the JSON root to be the list [[...], ...]
@app.post("/predict",
          response_model=PredictionOutput,
          summary="Predict Earthquake Epicenter")
def predict(
    # Expects the request body to be a JSON array: [[time,lat,lon], [time,lat,lon], ...]
    # If your Vercel app sends {"stations": [[...]]}, change this to:
    # stations_data: StationListInput = Body(..., embed=True, alias="stations")
    stations_data: StationListInput
    ):
    """
    Receives station arrival data (P-wave time, lat, lon) for at least 3 stations,
    performs feature engineering, predicts epicenter latitude and longitude.
    """
    global model
    if model is None:
        print("ERROR: Prediction attempt failed: Model is not loaded.", file=sys.stderr)
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

    print(f"INFO: Received prediction request with {len(stations_data)} stations.", file=sys.stderr)

    # Pydantic has already performed basic validation (list length >= 3, types, lat/lon ranges)
    # If you needed more complex validation beyond Pydantic, add it here.

    try:
        # --- Replicate Feature Engineering ---
        # Sort stations by arrival time just in case they aren't already
        sorted_stations = sorted(stations_data, key=lambda s: s.pWaveTime)

        first_arrival_time = sorted_stations[0].pWaveTime
        print(f"DEBUG: First arrival time: {first_arrival_time}", file=sys.stderr)

        time_diffs = []
        station_coords = []
        for station in sorted_stations:
            time_diff = station.pWaveTime - first_arrival_time
            time_diffs.append(time_diff)
            station_coords.extend([station.latitude, station.longitude])

        print(f"DEBUG: Time Diffs: {time_diffs}", file=sys.stderr)
        print(f"DEBUG: Station Coords: {station_coords}", file=sys.stderr)

        # Assemble features - limit to max 5 stations as in original script
        num_stations_to_use = min(len(sorted_stations), 5)
        features = []

        # Station 0 (Reference)
        features.extend([station_coords[0], station_coords[1]]) # lat0, lon0

        # Station 1
        if num_stations_to_use >= 2:
             features.extend([time_diffs[1], station_coords[2], station_coords[3]]) # timediff1, lat1, lon1

        # Station 2
        if num_stations_to_use >= 3:
             features.extend([station_coords[4], station_coords[5], time_diffs[2]]) # lat2, lon2, timediff2

        # Station 3
        if num_stations_to_use >= 4:
             features.extend([station_coords[6], station_coords[7], time_diffs[3]]) # lat3, lon3, timediff3

        # Station 4
        if num_stations_to_use >= 5:
             features.extend([station_coords[8], station_coords[9], time_diffs[4]]) # lat4, lon4, timediff4

        print(f"DEBUG: Assembled features ({len(features)}): {features}", file=sys.stderr)

        # Convert to NumPy array for the model
        input_array = np.array(features).reshape(1, -1)
        print(f"DEBUG: Input array shape: {input_array.shape}", file=sys.stderr)

        # --- Make prediction ---
        prediction = model.predict(input_array)
        print(f"DEBUG: Raw prediction from model: {prediction}", file=sys.stderr)

        # --- Format Output ---
        # Assuming prediction is [[latitude, longitude]]
        if prediction is not None and len(prediction) > 0 and len(prediction[0]) >= 2:
            pred_lat = float(prediction[0][0])
            pred_lon = float(prediction[0][1])
            print(f"INFO: Prediction successful - Lat: {pred_lat}, Lon: {pred_lon}", file=sys.stderr)
            return PredictionOutput(latitude=pred_lat, longitude=pred_lon)
        else:
            print("ERROR: Prediction result format unexpected.", file=sys.stderr)
            raise HTTPException(status_code=500, detail="Prediction failed: Unexpected model output format.")

    except HTTPException:
         raise # Re-raise validation or other specific HTTP exceptions
    except Exception as e:
        # Log the full error for debugging on the server
        import traceback
        print(f"ERROR: Unexpected error during prediction: {e}\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction.")

# Note: Uvicorn command for running locally is usually done via CMD in Dockerfile
# or terminal, not typically included directly here for deployment scenarios.