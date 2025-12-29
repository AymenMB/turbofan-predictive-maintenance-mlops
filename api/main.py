"""
FastAPI application for Turbofan RUL prediction.

This API serves predictions from the optimized XGBoost model.
Deployed model: model_optimized.ubj (RMSE: 50.71 cycles)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import deque
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predict Remaining Useful Life (RUL) of turbofan engines",
    version="1.1.0"
)

# Global model variable
model = None

# Sensors that were dropped during training (constant values)
DROPPED_SENSORS = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']

# Expected feature order (must match training data)
EXPECTED_FEATURES = [
    'setting_1', 'setting_2', 'setting_3',
    's_2', 's_3', 's_4', 's_6', 's_7', 's_8', 's_9',
    's_11', 's_12', 's_13', 's_14', 's_15',
    's_17', 's_20', 's_21'
]

# ========================================
# Monitoring & Drift Detection
# ========================================

# In-memory storage for recent predictions (last 100)
recent_predictions = deque(maxlen=100)

# Baseline statistics from training data (FD001 train set)
# These are approximate means calculated from the training dataset
BASELINE_STATS = {
    'setting_1': -0.0001,
    'setting_2': 0.0002,
    'setting_3': 100.0,
    's_2': 642.6,
    's_3': 1591.4,
    's_4': 1407.1,
    's_6': 21.6,
    's_7': 554.9,
    's_8': 2388.1,
    's_9': 9059.3,
    's_11': 47.5,
    's_12': 522.3,
    's_13': 2388.1,
    's_14': 8140.5,
    's_15': 8.44,
    's_17': 391.0,
    's_20': 39.1,
    's_21': 23.42
}

# Drift detection threshold (20% deviation)
DRIFT_THRESHOLD = 0.20


class EngineFeatures(BaseModel):
    """Input schema for engine sensor readings."""
    
    # Operational settings
    setting_1: float = Field(..., description="Operational setting 1")
    setting_2: float = Field(..., description="Operational setting 2")
    setting_3: float = Field(..., description="Operational setting 3")
    
    # Sensor measurements (all 21 sensors)
    s_1: float = Field(..., description="Sensor 1 (will be dropped)")
    s_2: float = Field(..., description="Sensor 2")
    s_3: float = Field(..., description="Sensor 3")
    s_4: float = Field(..., description="Sensor 4")
    s_5: float = Field(..., description="Sensor 5 (will be dropped)")
    s_6: float = Field(..., description="Sensor 6")
    s_7: float = Field(..., description="Sensor 7")
    s_8: float = Field(..., description="Sensor 8")
    s_9: float = Field(..., description="Sensor 9")
    s_10: float = Field(..., description="Sensor 10 (will be dropped)")
    s_11: float = Field(..., description="Sensor 11")
    s_12: float = Field(..., description="Sensor 12")
    s_13: float = Field(..., description="Sensor 13")
    s_14: float = Field(..., description="Sensor 14")
    s_15: float = Field(..., description="Sensor 15")
    s_16: float = Field(..., description="Sensor 16 (will be dropped)")
    s_17: float = Field(..., description="Sensor 17")
    s_18: float = Field(..., description="Sensor 18 (will be dropped)")
    s_19: float = Field(..., description="Sensor 19 (will be dropped)")
    s_20: float = Field(..., description="Sensor 20")
    s_21: float = Field(..., description="Sensor 21")

    class Config:
        schema_extra = {
            "example": {
                "setting_1": -0.0007,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70,
                "s_4": 1400.60, "s_5": 14.62, "s_6": 21.61,
                "s_7": 554.36, "s_8": 2388.06, "s_9": 9046.19,
                "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195,
                "s_16": 0.03, "s_17": 392, "s_18": 2388,
                "s_19": 100.0, "s_20": 39.06, "s_21": 23.4190
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for RUL prediction."""
    RUL: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    status: str = Field(..., description="Engine health status")
    confidence: str = Field(..., description="Prediction confidence level")


class MonitoringResponse(BaseModel):
    """Response schema for monitoring endpoint."""
    drift_detected: bool = Field(..., description="Whether data drift was detected")
    status: str = Field(..., description="Monitoring status message")
    metrics: Dict = Field(..., description="Drift detection metrics")
    recent_requests: int = Field(..., description="Number of recent requests analyzed")


@app.on_event("startup")
async def load_model():
    """Load the optimized XGBoost model on startup."""
    global model
    
    model_path = Path("model_optimized.ubj")
    
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    
    try:
        model = xgb.Booster()
        model.load_model(str(model_path))
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  Model type: XGBoost Booster")
        print(f"  Expected features: {len(EXPECTED_FEATURES)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Turbofan RUL Prediction API",
        "version": "1.1.0",
        "model": "XGBoost (Optimized with Optuna)",
        "performance": "RMSE: 50.71 cycles",
        "endpoints": {
            "predict": "POST /predict - Get RUL prediction",
            "health": "GET /health - Check API health",
            "monitoring": "GET /monitoring - Check for data drift",
            "docs": "GET /docs - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": "model_optimized.ubj"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(features: EngineFeatures):
    """
    Predict Remaining Useful Life (RUL) for a turbofan engine.
    
    Args:
        features: Engine sensor readings and operational settings
        
    Returns:
        PredictionResponse with RUL, status, and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Drop constant sensors (same as during training)
        input_data = input_data.drop(columns=DROPPED_SENSORS)
        
        # Ensure correct column order
        if list(input_data.columns) != EXPECTED_FEATURES:
            # Reorder columns to match training
            input_data = input_data[EXPECTED_FEATURES]
        
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(input_data)
        
        # Make prediction
        rul_pred = model.predict(dmatrix)[0]
        
        # Ensure non-negative RUL
        rul_pred = max(0.0, float(rul_pred))
        
        # Determine status based on RUL thresholds
        if rul_pred < 30:
            status = "Critical"
            confidence = "High"
        elif rul_pred < 80:
            status = "Warning"
            confidence = "Medium"
        else:
            status = "Healthy"
            confidence = "High"
        
        # Store input for drift monitoring (only processed features)
        input_record = {
            'timestamp': datetime.now().isoformat(),
            'features': input_data.iloc[0].to_dict(),
            'rul': rul_pred
        }
        recent_predictions.append(input_record)
        
        return PredictionResponse(
            RUL=round(rul_pred, 2),
            status=status,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Booster",
        "optimization": "Optuna (20 trials)",
        "performance": {
            "test_rmse": 50.71,
            "improvement_over_baseline": "1.26%"
        },
        "hyperparameters": {
            "learning_rate": 0.046,
            "max_depth": 3,
            "n_estimators": 287,
            "subsample": 0.969,
            "colsample_bytree": 0.782
        },
        "features": {
            "total_input": 24,
            "after_preprocessing": len(EXPECTED_FEATURES),
            "dropped_sensors": DROPPED_SENSORS
        }
    }


@app.get("/monitoring", response_model=MonitoringResponse)
async def monitor_drift():
    """
    Monitor data drift by comparing recent predictions with baseline statistics.
    
    Detects data drift when the mean of recent sensor readings deviates 
    more than 20% from the training data baseline.
    
    Returns:
        MonitoringResponse with drift detection status and metrics
    """
    if len(recent_predictions) == 0:
        return MonitoringResponse(
            drift_detected=False,
            status="No data available for monitoring",
            metrics={},
            recent_requests=0
        )
    
    # Calculate mean of recent sensor readings
    recent_features = [pred['features'] for pred in recent_predictions]
    recent_df = pd.DataFrame(recent_features)
    recent_means = recent_df.mean().to_dict()
    
    # Calculate deviation from baseline for each feature
    deviations = {}
    max_deviation = 0.0
    drifted_features = []
    
    for feature in EXPECTED_FEATURES:
        if feature in BASELINE_STATS and feature in recent_means:
            baseline_val = BASELINE_STATS[feature]
            recent_val = recent_means[feature]
            
            # Calculate percentage deviation
            if baseline_val != 0:
                deviation = abs(recent_val - baseline_val) / abs(baseline_val)
            else:
                deviation = abs(recent_val - baseline_val)
            
            deviations[feature] = {
                'baseline': round(baseline_val, 4),
                'recent': round(recent_val, 4),
                'deviation_pct': round(deviation * 100, 2)
            }
            
            # Track maximum deviation
            if deviation > max_deviation:
                max_deviation = deviation
            
            # Track features with significant drift
            if deviation > DRIFT_THRESHOLD:
                drifted_features.append({
                    'feature': feature,
                    'deviation_pct': round(deviation * 100, 2)
                })
    
    # Determine drift status
    drift_detected = max_deviation > DRIFT_THRESHOLD
    
    if drift_detected:
        status = f"Data Drift Warning - {len(drifted_features)} feature(s) exceed threshold"
    else:
        status = "No significant drift detected"
    
    # Prepare metrics
    metrics = {
        "max_deviation_pct": round(max_deviation * 100, 2),
        "threshold_pct": round(DRIFT_THRESHOLD * 100, 2),
        "drifted_features": drifted_features,
        "feature_statistics": deviations
    }
    
    return MonitoringResponse(
        drift_detected=drift_detected,
        status=status,
        metrics=metrics,
        recent_requests=len(recent_predictions)
    )


@app.get("/monitoring/reset")
async def reset_monitoring():
    """Reset monitoring data (clear recent predictions buffer)."""
    recent_predictions.clear()
    return {
        "status": "Monitoring data cleared",
        "recent_requests": 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
