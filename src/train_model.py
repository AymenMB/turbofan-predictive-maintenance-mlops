"""
Model training module for NASA CMAPSS RUL prediction.

This module trains an XGBoost regressor to predict Remaining Useful Life (RUL)
and logs all experiments to MLflow for tracking and reproducibility.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_preprocessing import load_and_process_data


def train_model(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42
):
    """
    Train XGBoost model for RUL prediction with MLflow tracking.
    
    This function:
    1. Loads and processes data
    2. Splits data by engine unit (time-series aware split)
    3. Trains XGBoost regressor
    4. Logs everything to MLflow
    5. Saves model artifact
    
    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Step size for gradient boosting
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility
    """
    
    # Set MLflow experiment
    mlflow.set_experiment("Turbofan_RUL_Prediction")
    
    print("=" * 70)
    print("TRAINING XGBOOST MODEL FOR RUL PREDICTION")
    print("=" * 70)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    df = load_and_process_data()
    
    # Time-series aware split: first 80 engines for training, last 20 for testing
    # This respects temporal dependencies and simulates real deployment
    print("\n[2/5] Splitting data (time-series aware)...")
    train_df = df[df['unit_nr'] <= 80].copy()
    test_df = df[df['unit_nr'] > 80].copy()
    
    print(f"  → Training set: {len(train_df)} samples from engines 1-80")
    print(f"  → Test set: {len(test_df)} samples from engines 81-100")
    
    # Prepare features and target
    # Drop identifiers (unit_nr, time_cycles) - they are not predictive features
    feature_cols = [col for col in df.columns 
                    if col not in ['unit_nr', 'time_cycles', 'RUL']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['RUL']
    X_test = test_df[feature_cols]
    y_test = test_df['RUL']
    
    print(f"  → Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")
    
    # Start MLflow run
    with mlflow.start_run():
        
        # Enable autologging for XGBoost
        mlflow.xgboost.autolog()
        
        print("\n[3/5] Training XGBoost model...")
        
        # Initialize and train model
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective='reg:squarederror',
            verbosity=1
        )
        
        model.fit(X_train, y_train)
        
        print("  ✓ Training complete!")
        
        # Make predictions
        print("\n[4/5] Evaluating model on test set...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  → Test RMSE: {rmse:.2f} cycles")
        print(f"  → Test MAE: {mae:.2f} cycles")
        print(f"  → R² Score: {r2:.4f}")
        
        # Log additional metrics manually
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        
        # Log dataset info
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("num_features", len(feature_cols))
        
        # Save model locally for API deployment
        print("\n[5/5] Saving model artifact...")
        model_path = project_root / "model.json"
        model.save_model(model_path)
        print(f"  ✓ Model saved to: {model_path}")
        
        # Log model as artifact in MLflow
        mlflow.log_artifact(str(model_path))
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE - Run logged to MLflow")
        print("=" * 70)
        print(f"\nFinal Test RMSE: {rmse:.2f} cycles")
        print(f"MLflow UI: Run 'mlflow ui' to view results")
        
        return model, rmse


if __name__ == "__main__":
    # Train the baseline model
    model, rmse = train_model(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("=" * 70)
    print(f"Test RMSE: {rmse:.2f} cycles")
    print("\nNext steps:")
    print("  1. Run 'mlflow ui' to view experiment tracking")
    print("  2. Try different hyperparameters for comparison")
    print("  3. Use Optuna for automated hyperparameter optimization")
