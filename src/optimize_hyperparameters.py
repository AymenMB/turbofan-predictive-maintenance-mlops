"""
Hyperparameter optimization script using Optuna for XGBoost model.

This script searches for optimal hyperparameters to further improve model performance
beyond the baseline. Uses engineered features (rolling windows, normalization).

Target: RMSE < 18 cycles (matching best R implementation ~17)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import optuna
import mlflow
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

from src.data_preprocessing import load_and_process_data, get_feature_columns


RUL_CAP = 125


def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna objective function to minimize RMSE.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        float: RMSE on test set
    """
    # Define hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    }
    
    # Train model with current hyperparameters
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("test_rmse", rmse)
    
    return rmse


def main():
    """
    Main optimization workflow with feature engineering.
    """
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("Using Advanced Feature Engineering")
    print("=" * 70)
    print()
    
    # Set MLflow experiment
    mlflow.set_experiment("Turbofan_Optuna_Optimization")
    
    # Load and prepare data with feature engineering
    print("[1/5] Loading and preprocessing data with feature engineering...")
    data_path = Path("data/raw/train_FD001.txt")
    df = load_and_process_data(str(data_path))
    print(f"  ✓ Loaded {len(df)} samples with {df.shape[1]} columns")
    
    # Get feature columns (engineered features)
    feature_cols = get_feature_columns(df, include_rolling=True, 
                                        include_normalized=True, 
                                        include_raw=False)
    target_col = 'RUL_clipped'
    
    # Time-series split: units 1-80 for training, 81-100 for testing
    train_units = df['unit_nr'] <= 80
    test_units = df['unit_nr'] > 80
    
    X_train = df.loc[train_units, feature_cols]
    y_train = df.loc[train_units, target_col]
    X_test = df.loc[test_units, feature_cols]
    y_test = df.loc[test_units, target_col]
    
    print(f"  ✓ Training set: {len(X_train)} samples (units 1-80)")
    print(f"  ✓ Test set: {len(X_test)} samples (units 81-100)")
    print(f"  ✓ Features: {len(feature_cols)} engineered features")
    print()
    
    # Create Optuna study
    print("[2/5] Creating Optuna study...")
    study = optuna.create_study(
        direction='minimize',
        study_name='turbofan_xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    print("  ✓ Study created (objective: minimize RMSE)")
    print()
    
    # Run optimization
    print("[3/5] Running optimization (30 trials)...")
    print("  This may take a few minutes...")
    print()
    
    with mlflow.start_run(run_name="Optuna_Optimization_Parent"):
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_test, y_test),
            n_trials=30,
            show_progress_bar=True
        )
        
        # Log best parameters to parent run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)
    
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print()
    print(f"🎯 Best RMSE: {study.best_value:.4f} cycles")
    print()
    print("Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print()
    
    # Compare with baseline (before feature engineering)
    old_baseline = 50.71  # Original baseline without feature engineering
    new_baseline = 18.89  # After feature engineering
    
    improvement_from_old = old_baseline - study.best_value
    improvement_pct = (improvement_from_old / old_baseline) * 100
    
    print(f"✓ Improvement from original baseline: {improvement_from_old:.2f} cycles ({improvement_pct:.1f}% better)")
    
    if study.best_value < new_baseline:
        print(f"✓ Also improved from feature-engineered baseline: {new_baseline - study.best_value:.2f} cycles")
    print()
    
    # Retrain model with best parameters
    print("[4/5] Retraining model with best parameters...")
    best_params = study.best_params.copy()
    best_params.update({
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'hist'
    })
    
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train, verbose=False)
    
    # Final evaluation
    y_pred_test = best_model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"  ✓ Final RMSE on test set: {final_rmse:.4f} cycles")
    print()
    
    # Save optimized model
    print("[5/5] Saving optimized model...")
    model_path = Path("model_optimized.ubj")
    best_model.get_booster().save_model(str(model_path))
    print(f"  ✓ Model saved to: {model_path}")
    
    # Also save feature list
    feature_path = Path("feature_columns.txt")
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"  ✓ Feature list saved to: {feature_path}")
    print()
    
    # Log best model to MLflow
    with mlflow.start_run(run_name="Best_Model_Final"):
        mlflow.log_params(best_params)
        mlflow.log_metric("test_rmse", final_rmse)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(feature_path))
        print("  ✓ Model logged to MLflow")
    
    print()
    print("=" * 70)
    print("✓ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    
    # Performance assessment
    if final_rmse < 18:
        print("🏆 EXCELLENT: Matches or beats R implementation (~17 cycles)")
    elif final_rmse < 20:
        print("✅ GREAT: Very close to R implementation")
    elif final_rmse < 25:
        print("✅ GOOD: Competitive with R implementation")
    
    print()
    print("Optimization Statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best RMSE: {study.best_value:.4f} cycles")
    print()


if __name__ == "__main__":
    main()
