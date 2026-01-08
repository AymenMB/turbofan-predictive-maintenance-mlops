"""
Model training step for ZenML pipeline.

Trains XGBoost regressor with engineered features and time-series aware split.
"""

import pandas as pd
import mlflow
from xgboost import XGBRegressor
from zenml import step
from zenml.client import Client


# Constants
RUL_CAP = 125


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns for training (rolling + normalized, not raw)."""
    exclude_cols = ['unit_nr', 'time_cycles', 'RUL', 'RUL_clipped']
    feature_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        # Include rolling features, normalized features, and settings
        is_rolling = col.endswith('_mean') or col.endswith('_std')
        is_normalized = col.endswith('_norm')
        is_setting = col.startswith('setting_')
        
        if is_rolling or is_normalized or is_setting:
            feature_cols.append(col)
    
    return sorted(feature_cols)


@step(enable_cache=False)
def train_model(
    df: pd.DataFrame,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> XGBRegressor:
    """
    Train XGBoost model for RUL prediction with engineered features.
    
    Args:
        df: Cleaned DataFrame with engineered features and RUL_clipped target
        n_estimators: Number of boosting rounds
        learning_rate: Step size for gradient boosting
        max_depth: Maximum tree depth
        subsample: Fraction of samples per tree
        colsample_bytree: Fraction of features per tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    print("=" * 70)
    print("TRAINING XGBOOST MODEL (ZenML Pipeline)")
    print("With Advanced Feature Engineering")
    print("=" * 70)
    
    # Time-series aware split: first 80 engines for training, last 20 for testing
    print("\n[1/3] Splitting data (time-series aware)...")
    train_df = df[df['unit_nr'] <= 80].copy()
    
    print(f"  → Training set: {len(train_df)} samples from engines 1-80")
    
    # Get engineered feature columns
    feature_cols = get_feature_columns(df)
    
    # Use clipped RUL as target
    target_col = 'RUL_clipped'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    print(f"  → Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (first 5)")
    print(f"  → Target: {target_col} (capped at {RUL_CAP} cycles)")
    
    # Train model
    print("\n[2/3] Training XGBoost model...")
    print(f"  → n_estimators: {n_estimators}")
    print(f"  → learning_rate: {learning_rate}")
    print(f"  → max_depth: {max_depth}")
    
    # Set MLflow experiment (ZenML integrates with MLflow)
    try:
        experiment_tracker = Client().active_stack.experiment_tracker
        if experiment_tracker:
            mlflow.set_experiment("Turbofan_RUL_Prediction")
    except Exception:
        pass  # MLflow not configured
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        objective='reg:squarederror',
        tree_method='hist'
    )
    
    model.fit(X_train, y_train)
    
    print("  ✓ Training complete!")
    print(f"  → Model parameters: n_estimators={n_estimators}, lr={learning_rate}, depth={max_depth}")
    
    print("\n[3/3] Model ready for evaluation")
    
    return model
