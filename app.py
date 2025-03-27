# --- File: earthquake-prediction-api/app.py ---
# MODIFIED TO ACCEPT ARRAY OF ARRAYS

import os
import joblib
import numpy as np
# pandas might not be needed if you build the numpy array directly
# import pandas as pd
from fastapi import FastAPI, HTTPException, Body # Body might not be needed now
from pydantic import BaseModel, Field # Still use for OUTPUT validation
from typing import List, Any # Change input type hint
import uvicorn
import sys

# --- Define ONLY the Output Data Structure ---
# We won't use Pydantic for INPUT validation in this approach
class PredictionOutput(BaseModel):
    latitude: float = Field(..., description="Predicted epicenter latitude")
    longitude: float = Field(..., description="Predicted epicenter longitude")

# --- Model Loading (Remains the same) ---
MODEL_FILENAME = "XGBoost_model.pkl"
model = None
app = FastAPI(title="Earthquake Epicenter Prediction API")

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    print(f"INFO: Attempting to load model from: {model_path}", file=sys.stderr)
    if not os.path.exists(model_path):
        print(f"FATAL ERROR: Model file not found at {model_path}", file=sys.stderr)
        return
    try:
        model = joblib.load(model_path)
        print(f"INFO: Model loaded successfully. Type: {type(model)}", file=sys.stderr)
    except Exception as e:
        print(f"FATAL ERROR: Error loading model from {model_path}: {e}", file=sys.stderr)
        model = None

# --- API Endpoints (Root and Health remain the same) ---
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "model_loaded": model is not None}

@app.route('/healthz') # Keep this route name consistent
def health_check():
    if model is not None:
        return "OK", 200
    else:
        return "Model not loaded", 503

# --- MODIFIED PREDICT ENDPOINT ---
@app.post("/predict",
          response_model=PredictionOutput, # Still validate the output format
          summary="Predict Earthquake Epicenter (Accepts Array of Arrays)")
def predict(
    # Change the type hint to expect a List containing inner Lists (of Any type initially)
    stations_data: List[List[Any]]
    ):
    global model
    if model is None:
        print("ERROR: Prediction attempt failed: Model is not loaded.", file=sys.stderr)
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load.")

    print(f"INFO: Received prediction request with {len(stations_data)} stations (array of arrays format).", file=sys.stderr)
    # print(f"DEBUG: Raw input data: {stations_data}", file=sys.stderr) # Uncomment for deep debugging

    # --- MANUAL INPUT VALIDATION ---
    # 1. Check minimum length
    if len(stations_data) < 3:
        print(f"ERROR: Validation failed - received {len(stations_data)} stations, require at least 3.", file=sys.stderr)
        raise HTTPException(status_code=422, detail="Input must contain data for at least 3 stations.")

    # 2. Validate each inner array
    validated_stations = []
    for i, station_array in enumerate(stations_data):
        if not isinstance(station_array, list) or len(station_array) != 3:
            print(f"ERROR: Validation failed - station {i} is not an array of length 3.", file=sys.stderr)
            raise HTTPException(status_code=422, detail=f"Station data at index {i} must be an array of [time, latitude, longitude].")

        time, lat, lon = station_array
        # Check types (attempt conversion)
        try:
            time = float(time)
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
             print(f"ERROR: Validation failed - station {i} contains non-numeric data.", file=sys.stderr)
             raise HTTPException(status_code=422, detail=f"Station data at index {i} must contain numeric values for time, latitude, longitude.")

        # Check ranges
        if not (-90 <= lat <= 90):
            print(f"ERROR: Validation failed - station {i} has invalid latitude ({lat}).", file=sys.stderr)
            raise HTTPException(status_code=422, detail=f"Invalid latitude for station at index {i}: must be between -90 and 90.")
        if not (-180 <= lon <= 180):
            print(f"ERROR: Validation failed - station {i} has invalid longitude ({lon}).", file=sys.stderr)
            raise HTTPException(status_code=422, detail=f"Invalid longitude for station at index {i}: must be between -180 and 180.")

        validated_stations.append([time, lat, lon]) # Store validated numeric data

    # --- FEATURE ENGINEERING (using validated data) ---
    try:
        # Sort stations by arrival time (index 0)
        sorted_stations = sorted(validated_stations, key=lambda s: s[0])

        first_arrival_time = sorted_stations[0][0]
        print(f"DEBUG: First arrival time: {first_arrival_time}", file=sys.stderr)

        time_diffs = []
        station_coords = []
        for station in sorted_stations:
            time_diff = station[0] - first_arrival_time
            time_diffs.append(time_diff)
            station_coords.extend([station[1], station[2]]) # Lat, Lon

        print(f"DEBUG: Time Diffs: {time_diffs}", file=sys.stderr)
        print(f"DEBUG: Station Coords: {station_coords}", file=sys.stderr)

        # Assemble features - limit to max 5 stations
        num_stations_to_use = min(len(sorted_stations), 5)
        features = []

        # Station 0 (Reference) - Lat, Lon (Indices 0, 1 in station_coords)
        features.extend([station_coords[0], station_coords[1]])

        # Station 1 - TimeDiff, Lat, Lon (Indices 1 in time_diffs, 2, 3 in station_coords)
        if num_stations_to_use >= 2:
             features.extend([time_diffs[1], station_coords[2], station_coords[3]])

        # Station 2 - Lat, Lon, TimeDiff (Indices 4, 5 in station_coords, 2 in time_diffs)
        if num_stations_to_use >= 3:
             features.extend([station_coords[4], station_coords[5], time_diffs[2]])

        # Station 3 - Lat, Lon, TimeDiff (Indices 6, 7 in station_coords, 3 in time_diffs)
        if num_stations_to_use >= 4:
             features.extend([station_coords[6], station_coords[7], time_diffs[3]])

        # Station 4 - Lat, Lon, TimeDiff (Indices 8, 9 in station_coords, 4 in time_diffs)
        if num_stations_to_use >= 5:
             features.extend([station_coords[8], station_coords[9], time_diffs[4]])

        print(f"DEBUG: Assembled features ({len(features)}): {features}", file=sys.stderr)

        # Convert to NumPy array for the model
        input_array = np.array(features).reshape(1, -1)
        print(f"DEBUG: Input array shape: {input_array.shape}", file=sys.stderr)

        # --- Make prediction ---
        prediction = model.predict(input_array)
        print(f"DEBUG: Raw prediction from model: {prediction}", file=sys.stderr)

        # --- Format Output ---
        if prediction is not None and len(prediction) > 0 and len(prediction[0]) >= 2:
            pred_lat = float(prediction[0][0])
            pred_lon = float(prediction[0][1])
            print(f"INFO: Prediction successful - Lat: {pred_lat}, Lon: {pred_lon}", file=sys.stderr)
            # Return data matching the PredictionOutput model
            return PredictionOutput(latitude=pred_lat, longitude=pred_lon)
        else:
            print("ERROR: Prediction result format unexpected.", file=sys.stderr)
            raise HTTPException(status_code=500, detail="Prediction failed: Unexpected model output format.")

    except HTTPException:
         raise # Re-raise specific validation HTTP exceptions
    except Exception as e:
        import traceback
        print(f"ERROR: Unexpected error during prediction: {e}\n{traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction.")


# --- Run App Section (Remains the same for local testing) ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"--- Starting FastAPI server on host 0.0.0.0 port {port} ---")
    # Run with uvicorn from terminal for development: uvicorn app:app --reload --host 0.0.0.0 --port 5000
    # This block might not be hit when running with gunicorn/uvicorn via Procfile
    uvicorn.run(app, host='0.0.0.0', port=port) # Use uvicorn.run if running directly