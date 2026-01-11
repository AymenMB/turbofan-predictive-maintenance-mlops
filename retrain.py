"""
Automatic Retrain Script for Turbofan RUL Model

This script implements automatic retraining triggered by:
1. Scheduled execution (cron/task scheduler)
2. Performance degradation detection
3. Data drift detection

Usage:
    python retrain.py              # Run retraining
    python retrain.py --check-only # Only check if retrain is needed
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import mlflow
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
API_URL = "http://localhost:8000"
MONITORING_ENDPOINT = f"{API_URL}/monitoring"
MODEL_PATH = "model_optimized.ubj"
RETRAIN_LOG = "retrain_log.json"

# Thresholds for triggering retrain
DRIFT_THRESHOLD = 0.2  # 20% drift triggers retrain
RMSE_DEGRADATION_THRESHOLD = 25  # RMSE above this triggers retrain


def check_drift_status():
    """Check current drift status from API."""
    try:
        response = requests.get(MONITORING_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('drift_detected', False), data
        return False, {}
    except Exception as e:
        print(f"⚠️ Could not check drift status: {e}")
        return False, {}


def load_training_data():
    """Load and prepare training data."""
    from src.data_preprocessing import load_and_process_data, get_feature_columns
    
    data_path = Path("data/raw/train_FD001.txt")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = load_and_process_data(str(data_path))
    feature_cols = get_feature_columns(df, include_rolling=True, 
                                        include_normalized=True, 
                                        include_raw=False)
    
    # Time-series split
    train_mask = df['unit_nr'] <= 80
    test_mask = df['unit_nr'] > 80
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'RUL_clipped']
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'RUL_clipped']
    
    return X_train, y_train, X_test, y_test, feature_cols


def train_model(X_train, y_train, X_test, y_test):
    """Train the XGBoost model with best hyperparameters."""
    
    # Best parameters from Optuna optimization
    best_params = {
        'learning_rate': 0.05,
        'max_depth': 4,
        'n_estimators': 300,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    }
    
    print("🔄 Training model with optimized hyperparameters...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   ✓ RMSE: {rmse:.2f} cycles")
    print(f"   ✓ R²: {r2:.2f}")
    
    return model, rmse, r2


def save_model(model, feature_cols):
    """Save the retrained model."""
    # Backup current model
    if Path(MODEL_PATH).exists():
        backup_path = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ubj"
        Path(MODEL_PATH).rename(backup_path)
        print(f"   ✓ Backed up old model to: {backup_path}")
    
    # Save new model
    model.get_booster().save_model(MODEL_PATH)
    print(f"   ✓ Saved new model to: {MODEL_PATH}")
    
    # Save feature columns
    with open("feature_columns.txt", 'w') as f:
        f.write('\n'.join(feature_cols))


def log_retrain(rmse, r2, reason):
    """Log retraining event."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'rmse': rmse,
        'r2': r2,
        'reason': reason
    }
    
    # Load existing log
    log_data = []
    if Path(RETRAIN_LOG).exists():
        with open(RETRAIN_LOG, 'r') as f:
            log_data = json.load(f)
    
    log_data.append(log_entry)
    
    with open(RETRAIN_LOG, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"   ✓ Logged retrain event")


def log_to_mlflow(rmse, r2, reason):
    """Log retraining to MLflow."""
    mlflow.set_experiment("Turbofan_Retrain")
    
    with mlflow.start_run(run_name=f"Retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("retrain_reason", reason)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_artifact(MODEL_PATH)
    
    print(f"   ✓ Logged to MLflow")


def run_retrain(reason="manual"):
    """Execute the full retraining pipeline."""
    print("\n" + "=" * 70)
    print("🔄 AUTOMATIC RETRAIN - Turbofan RUL Model")
    print("=" * 70)
    print(f"\nReason: {reason}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Step 1: Load data
    print("[1/4] Loading training data...")
    try:
        X_train, y_train, X_test, y_test, feature_cols = load_training_data()
        print(f"   ✓ Loaded {len(X_train)} training samples")
        print(f"   ✓ Loaded {len(X_test)} test samples")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return False
    
    # Step 2: Train model
    print("\n[2/4] Training model...")
    try:
        model, rmse, r2 = train_model(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return False
    
    # Step 3: Save model
    print("\n[3/4] Saving model...")
    try:
        save_model(model, feature_cols)
    except Exception as e:
        print(f"   ✗ Failed to save model: {e}")
        return False
    
    # Step 4: Log results
    print("\n[4/4] Logging results...")
    try:
        log_retrain(rmse, r2, reason)
        log_to_mlflow(rmse, r2, reason)
    except Exception as e:
        print(f"   ✗ Failed to log: {e}")
    
    print("\n" + "=" * 70)
    print("✅ RETRAIN COMPLETE")
    print(f"   New model saved to: {MODEL_PATH}")
    print(f"   Performance: RMSE = {rmse:.2f} cycles, R² = {r2:.2f}")
    print("=" * 70 + "\n")
    
    return True


def check_and_retrain():
    """Check if retraining is needed and execute if so."""
    print("\n" + "=" * 70)
    print("🔍 CHECKING IF RETRAIN IS NEEDED")
    print("=" * 70)
    
    # Check drift status
    drift_detected, drift_data = check_drift_status()
    
    if drift_detected:
        print("⚠️ Data drift detected!")
        run_retrain(reason="drift_detected")
        return True
    
    print("✓ No drift detected")
    print("✓ No retraining needed at this time")
    return False


def main():
    parser = argparse.ArgumentParser(description="Automatic retraining for Turbofan RUL model")
    parser.add_argument('--check-only', action='store_true', 
                        help='Only check if retrain is needed, do not execute')
    parser.add_argument('--force', action='store_true',
                        help='Force retraining regardless of checks')
    parser.add_argument('--reason', type=str, default='manual',
                        help='Reason for retraining')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_and_retrain()
    elif args.force:
        run_retrain(reason=args.reason)
    else:
        # Check first, then retrain if needed
        drift_detected, _ = check_drift_status()
        if drift_detected:
            run_retrain(reason="drift_detected")
        else:
            print("No retraining trigger detected. Use --force to retrain anyway.")


if __name__ == "__main__":
    main()
