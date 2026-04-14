import os
import pickle
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Constants
MODEL_PATH = "tpot_best_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
CLEANED_DATA_PATH = "cleaned_traffic_data.csv"
DATASET_PATH = "Banglore_traffic_Dataset.csv"

# Globals
model = None
encoders = {}
default_row = {}

def load_resources():
    global model, encoders, default_row
    
    # Check paths
    actual_model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "C:\\Advanced projects\\Bangalore_Traffic\\tpot_best_model.pkl"
    actual_encoders_path = ENCODERS_PATH if os.path.exists(ENCODERS_PATH) else "C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"
    
    try:
        with open(actual_model_path, "rb") as f:
            model = pickle.load(f)
            print("Loaded TPOT pipeline successfully from", actual_model_path)
    except Exception as e:
        print(f"Warning: No valid model found at {actual_model_path}. Error: {e}")

    try:
        with open(actual_encoders_path, "rb") as f:
            encoders = pickle.load(f)
            print("Loaded Encoders successfully.")
    except Exception as e:
        print(f"Warning: No valid encoders found at {actual_encoders_path}. Error: {e}")
        
    try:
        if os.path.exists(CLEANED_DATA_PATH):
            df_clean = pd.read_csv(CLEANED_DATA_PATH)
        elif os.path.exists(DATASET_PATH):
            df_clean = pd.read_csv(DATASET_PATH)
        else:
            df_clean = pd.DataFrame()
            
        if not df_clean.empty:
            df_features = df_clean.drop(columns=["Traffic_Volume", "Date"], errors='ignore')
            default_row = df_features.mean(numeric_only=True).to_dict()  # Numeric averages
            # Keep categorical modes if numeric failed
            for col in df_features.select_dtypes(include=['object']):
                default_row[col] = df_features[col].mode()[0] if not df_features[col].mode().empty else ""
                
            print("Loaded default values for inference gracefully.")
        else:
            # Fallback hardcoded defaults if files cannot be read
            default_row = {
                "Average Speed": 40.0,
                "Travel Time Index": 1.2,
                "Congestion Level": 70.0,
                "Road Capacity Utilization": 85.0,
                "Incident Reports": 1,
                "Environmental Impact": 100.0,
                "Public Transport Usage": 40.0,
                "Parking Usage": 80.0,
                "Pedestrian and Cyclist Count": 100
            }
            print("Used hardcoded fallback default data row.")
    except Exception as e:
        print("Could not load default data row:", e)

load_resources()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    Area_Name: str = Field(alias="Area Name")
    Road_Intersection_Name: str = Field(alias="Road/Intersection Name")
    Weather_Conditions: str = Field(alias="Weather Conditions")
    Traffic_Signal_Compliance: float = Field(alias="Traffic Signal Compliance")
    Roadwork_and_Construction_Activity: str = Field(alias="Roadwork and Construction Activity")

@app.post("/predict")
def predict_traffic(req: PredictionRequest):
    if model is None:
        return {"error": "Model uninitialized. Please ensure the pipeline file exists."}

    user_inputs = {
        "Area Name": req.Area_Name,
        "Road/Intersection Name": req.Road_Intersection_Name,
        "Weather Conditions": req.Weather_Conditions,
        "Roadwork and Construction Activity": req.Roadwork_and_Construction_Activity,
    }

    encoded_vals = {}
    for col, val in user_inputs.items():
        if col in encoders:
            enc = encoders[col]
            classes = list(enc.classes_)
            if val in classes:
                encoded_vals[col] = enc.transform([val])[0]
            else:
                encoded_vals[col] = enc.transform([classes[0]])[0] 
        else:
            encoded_vals[col] = val

    final_row = dict(default_row)
    for k, v in encoded_vals.items():
        final_row[k] = v
        
    final_row["Traffic Signal Compliance"] = req.Traffic_Signal_Compliance
    
    try:
        df_pred = pd.DataFrame([final_row])
        # Reorder to match expected columns
        if hasattr(model, 'feature_names_in_'):
            expected_feats = [feat for feat in model.feature_names_in_ if feat in df_pred.columns]
            df_pred = df_pred[expected_feats]
        
        pred = model.predict(df_pred)
        return {"predicted_volume": int(pred[0])}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
def serve_index():
    return FileResponse("index.html")

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Make sure to run inside the current folder
    uvicorn.run("fastapi_server:app", host="127.0.0.1", port=8000, reload=True)
