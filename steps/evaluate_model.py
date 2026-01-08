"""
Model evaluation step for ZenML pipeline.

Evaluates trained model with engineered features and logs metrics.
"""

import pandas as pd
import numpy as np
import mlflow
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from zenml import step
from zenml.client import Client


RUL_CAP = 125


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns for training (rolling + normalized, not raw)."""
    exclude_cols = ['unit_nr', 'time_cycles', 'RUL', 'RUL_clipped']
    feature_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        is_rolling = col.endswith('_mean') or col.endswith('_std')
        is_normalized = col.endswith('_norm')
        is_setting = col.startswith('setting_')
        
        if is_rolling or is_normalized or is_setting:
            feature_cols.append(col)
    
    return sorted(feature_cols)


@step
def evaluate_model(
    model: XGBRegressor,
    df: pd.DataFrame
) -> dict:
    """
    Evaluate trained model on test set and log metrics.
    
    Args:
        model: Trained XGBoost model
        df: Cleaned DataFrame with engineered features and RUL_clipped target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("=" * 70)
    print("EVALUATING MODEL (ZenML Pipeline)")
    print("=" * 70)
    
    # Prepare test data (engines 81-100)
    print("\n[1/3] Preparing test data...")
    test_df = df[df['unit_nr'] > 80].copy()
    
    # Get engineered feature columns
    feature_cols = get_feature_columns(df)
    
    # Use clipped RUL as target
    X_test = test_df[feature_cols]
    y_test = test_df['RUL_clipped']
    
    print(f"  → Test set: {len(test_df)} samples from engines 81-100")
    print(f"  → Using {len(feature_cols)} engineered features")
    
    # Make predictions
    print("\n[2/3] Computing predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_r2": float(r2)
    }
    
    print(f"  → Test RMSE: {rmse:.2f} cycles")
    print(f"  → Test MAE: {mae:.2f} cycles")
    print(f"  → R² Score: {r2:.4f}")
    
    # Performance assessment
    if rmse < 20:
        print("✅ EXCELLENT: Matches R implementation (~17 cycles)")
    elif rmse < 25:
        print("✅ GOOD: Close to R implementation (< 25 cycles)")
    else:
        print("⚠️  MODERATE: Room for improvement")
    
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
            print("  ⚠ No experiment tracker configured")
    except Exception as e:
        print(f"  ⚠ Could not log to MLflow: {e}")
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    
    return metrics
