"""
Model training step for ZenML pipeline.

Trains XGBoost regressor with time-series aware split.
"""

import pandas as pd
import mlflow
from xgboost import XGBRegressor
from zenml import step
from zenml.client import Client


@step(enable_cache=False)
def train_model(
    df: pd.DataFrame,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 42
) -> XGBRegressor:
    """
    Train XGBoost model for RUL prediction with time-series split.
    
    Args:
        df: Cleaned DataFrame with RUL target
        n_estimators: Number of boosting rounds
        learning_rate: Step size for gradient boosting
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    print("=" * 70)
    print("TRAINING XGBOOST MODEL (ZenML Pipeline)")
    print("=" * 70)
    
    # Time-series aware split: first 80 engines for training, last 20 for testing
    print("\n[1/3] Splitting data (time-series aware)...")
    train_df = df[df['unit_nr'] <= 80].copy()
    test_df = df[df['unit_nr'] > 80].copy()
    
    print(f"  → Training set: {len(train_df)} samples from engines 1-80")
    print(f"  → Test set: {len(test_df)} samples from engines 81-100")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns 
                    if col not in ['unit_nr', 'time_cycles', 'RUL']]
    
    X_train = train_df[feature_cols]
    y_train = train_df['RUL']
    
    print(f"  → Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (first 5)")
    
    # Train model
    print("\n[2/3] Training XGBoost model...")
    
    # Set MLflow experiment (ZenML integrates with MLflow)
    experiment_tracker = Client().active_stack.experiment_tracker
    if experiment_tracker:
        mlflow.set_experiment("Turbofan_RUL_Prediction")
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    print("  ✓ Training complete!")
    print(f"  → Model parameters: n_estimators={n_estimators}, lr={learning_rate}, depth={max_depth}")
    
    print("\n[3/3] Model ready for evaluation")
    
    return model
