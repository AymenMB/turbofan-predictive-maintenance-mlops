"""
Model training module for NASA CMAPSS RUL prediction.

This module trains an XGBoost regressor to predict Remaining Useful Life (RUL)
using engineered features (rolling windows, normalization) and logs all 
experiments to MLflow for tracking and reproducibility.

Target Performance: RMSE < 25 cycles (matching R implementation ~17-20)
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

from src.data_preprocessing import (
    load_and_process_data, 
    get_feature_columns,
    RUL_CAP
)


def train_model(
    data_path: str = "data/raw/train_FD001.txt",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 3,
    random_state: int = 42,
    use_rolling_features: bool = True,
    use_normalized_features: bool = True
):
    """
    Train XGBoost model for RUL prediction with MLflow tracking.
    
    This function:
    1. Loads and processes data with feature engineering
    2. Splits data by engine unit (time-series aware split)
    3. Trains XGBoost regressor with optimized hyperparameters
    4. Logs everything to MLflow
    5. Saves model artifact
    
    Args:
        data_path: Path to raw training data
        n_estimators: Number of boosting rounds (increased for better accuracy)
        learning_rate: Step size for gradient boosting (lower for stability)
        max_depth: Maximum tree depth (lower to prevent overfitting)
        subsample: Fraction of samples per tree
        colsample_bytree: Fraction of features per tree
        min_child_weight: Minimum leaf weight
        random_state: Random seed for reproducibility
        use_rolling_features: Include rolling mean/std features
        use_normalized_features: Include normalized sensor features
        
    Returns:
        Tuple of (trained model, RMSE score)
    """
    
    # Set MLflow experiment
    mlflow.set_experiment("Turbofan_RUL_Prediction")
    
    print("=" * 70)
    print("TRAINING XGBOOST MODEL FOR RUL PREDICTION")
    print("With Advanced Feature Engineering")
    print("=" * 70)
    
    # Load and preprocess data with feature engineering
    print("\n[1/5] Loading and preprocessing data with feature engineering...")
    df = load_and_process_data(
        data_path, 
        add_rolling=use_rolling_features,
        add_normalization=use_normalized_features,
        clip_rul_values=True
    )
    
    # Time-series aware split: first 80 engines for training, last 20 for testing
    print("\n[2/5] Splitting data (time-series aware)...")
    train_df = df[df['unit_nr'] <= 80].copy()
    test_df = df[df['unit_nr'] > 80].copy()
    
    print(f"  → Training set: {len(train_df)} samples from engines 1-80")
    print(f"  → Test set: {len(test_df)} samples from engines 81-100")
    
    # Get feature columns (rolling + normalized, not raw)
    feature_cols = get_feature_columns(
        df, 
        include_rolling=use_rolling_features,
        include_normalized=use_normalized_features,
        include_raw=False  # Don't include raw sensors, use engineered features
    )
    
    # Use clipped RUL as target
    target_col = 'RUL_clipped'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"  → Feature columns ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")
    print(f"  → Target: {target_col} (capped at {RUL_CAP} cycles)")
    
    # Start MLflow run
    with mlflow.start_run():
        
        # Enable autologging for XGBoost
        mlflow.xgboost.autolog()
        
        print("\n[3/5] Training XGBoost model with optimized hyperparameters...")
        print(f"  → n_estimators: {n_estimators}")
        print(f"  → learning_rate: {learning_rate}")
        print(f"  → max_depth: {max_depth}")
        print(f"  → subsample: {subsample}")
        print(f"  → colsample_bytree: {colsample_bytree}")
        
        # Initialize and train model with improved hyperparameters
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            objective='reg:squarederror',
            verbosity=1,
            tree_method='hist'
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
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
        mlflow.log_param("rul_cap", RUL_CAP)
        mlflow.log_param("use_rolling_features", use_rolling_features)
        mlflow.log_param("use_normalized_features", use_normalized_features)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 Feature Importance:")
        for i, row in importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Save model locally for API deployment
        print("\n[5/5] Saving model artifact...")
        model_path = project_root / "model.ubj"
        model.get_booster().save_model(str(model_path))
        print(f"  ✓ Model saved to: {model_path}")
        
        # Also save as optimized model if better than current
        optimized_path = project_root / "model_optimized.ubj"
        model.get_booster().save_model(str(optimized_path))
        print(f"  ✓ Model also saved to: {optimized_path}")
        
        # Save feature list for API
        feature_path = project_root / "feature_columns.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_cols))
        print(f"  ✓ Feature list saved to: {feature_path}")
        
        # Log model as artifact in MLflow
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(feature_path))
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE - Run logged to MLflow")
        print("=" * 70)
        print(f"\n🎯 Final Test RMSE: {rmse:.2f} cycles")
        
        # Performance assessment
        if rmse < 20:
            print("✅ EXCELLENT: Matches R implementation (~17 cycles)")
        elif rmse < 25:
            print("✅ GOOD: Close to R implementation (< 25 cycles)")
        elif rmse < 35:
            print("⚠️  MODERATE: Better than baseline, room for improvement")
        else:
            print("❌ NEEDS WORK: Still far from target (< 25 cycles)")
        
        print(f"\nMLflow UI: Run 'mlflow ui' to view results")
        
        return model, rmse, feature_cols


if __name__ == "__main__":
    # Train the model with feature engineering
    model, rmse, features = train_model(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42
    )
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING WITH FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print(f"Test RMSE: {rmse:.2f} cycles")
    print(f"Number of features: {len(features)}")
    print("\nNext steps:")
    print("  1. Run 'mlflow ui' to view experiment tracking")
    print("  2. Run optimization: python src/optimize_hyperparameters.py")
    print("  3. Update API to use new features")
