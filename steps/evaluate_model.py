"""
Model evaluation step for ZenML pipeline.

Evaluates trained model and logs metrics.
"""

import pandas as pd
import numpy as np
import mlflow
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from zenml import step
from zenml.client import Client


@step
def evaluate_model(
    model: XGBRegressor,
    df: pd.DataFrame
) -> dict:
    """
    Evaluate trained model on test set and log metrics.
    
    Args:
        model: Trained XGBoost model
        df: Cleaned DataFrame with RUL target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("=" * 70)
    print("EVALUATING MODEL (ZenML Pipeline)")
    print("=" * 70)
    
    # Prepare test data (engines 81-100)
    print("\n[1/3] Preparing test data...")
    test_df = df[df['unit_nr'] > 80].copy()
    
    feature_cols = [col for col in df.columns 
                    if col not in ['unit_nr', 'time_cycles', 'RUL']]
    
    X_test = test_df[feature_cols]
    y_test = test_df['RUL']
    
    print(f"  → Test set: {len(test_df)} samples from engines 81-100")
    
    # Make predictions
    print("\n[2/3] Computing predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2
    }
    
    print(f"  → Test RMSE: {rmse:.2f} cycles")
    print(f"  → Test MAE: {mae:.2f} cycles")
    print(f"  → R² Score: {r2:.4f}")
    
    # Log metrics to MLflow (if experiment tracker is configured)
    print("\n[3/3] Logging metrics...")
    try:
        experiment_tracker = Client().active_stack.experiment_tracker
        if experiment_tracker:
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            print("  ✓ Metrics logged to MLflow")
        else:
            print("  ⚠ No experiment tracker configured, metrics not logged")
    except Exception as e:
        print(f"  ⚠ Could not log to MLflow: {e}")
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    
    return metrics
