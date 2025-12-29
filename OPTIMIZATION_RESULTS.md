# Hyperparameter Optimization Results

## Summary

Successfully implemented Optuna-based hyperparameter optimization for the XGBoost model, achieving **1.26% improvement** over the baseline.

## Performance Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **RMSE** | 51.35 cycles | **50.71 cycles** | **0.64 cycles (1.26%)** |
| MAE | 36.55 cycles | Not measured | - |
| R² Score | 0.5609 | Not measured | - |

## Optimization Configuration

**Framework**: Optuna 4.6.0  
**Algorithm**: Tree-structured Parzen Estimator (TPE)  
**Trials**: 20  
**Best Trial**: #11  
**MLflow Experiment**: `Turbofan_Optuna_Optimization`

## Best Hyperparameters Found

```python
{
    'learning_rate': 0.046410361893496764,  # Lower than baseline (0.1)
    'max_depth': 3,                         # Shallower trees (baseline: 6)
    'n_estimators': 287,                    # More trees (baseline: 100)
    'subsample': 0.9687557027925864,        # High row sampling
    'colsample_bytree': 0.7824927945666315, # Moderate column sampling
    'min_child_weight': 4,                  # Regularization
    'gamma': 0.9974770960844059,            # Minimum loss reduction
    'reg_alpha': 2.1355615281666593,        # L1 regularization
    'reg_lambda': 2.286422657069489         # L2 regularization
}
```

## Key Insights

### 1. **Lower Learning Rate**
- Optimized: `0.046` vs Baseline: `0.1`
- Smaller steps lead to better generalization

### 2. **Shallower Trees**
- Optimized: `max_depth=3` vs Baseline: `max_depth=6`
- Prevents overfitting on time-series data

### 3. **More Trees**
- Optimized: `287` vs Baseline: `100`
- Compensates for lower learning rate

### 4. **Strong Regularization**
- Added L1 (`reg_alpha=2.14`) and L2 (`reg_lambda=2.29`) regularization
- Prevents overfitting on engine-specific patterns

## Optimization Search Space

| Parameter | Min | Max | Type |
|-----------|-----|-----|------|
| learning_rate | 0.01 | 0.3 | float |
| max_depth | 3 | 10 | int |
| n_estimators | 50 | 300 | int |
| subsample | 0.6 | 1.0 | float |
| colsample_bytree | 0.6 | 1.0 | float |
| min_child_weight | 1 | 10 | int |
| gamma | 0.0 | 5.0 | float |
| reg_alpha | 0.0 | 5.0 | float |
| reg_lambda | 0.0 | 5.0 | float |

## Trial History (Top 5)

| Trial | RMSE (cycles) | learning_rate | max_depth | n_estimators |
|-------|---------------|---------------|-----------|--------------|
| **#11** | **50.7053** | 0.0464 | 3 | 287 |
| #12 | 50.7638 | 0.0483 | 3 | 288 |
| #15 | 50.9558 | 0.0547 | 4 | 261 |
| #4 | 51.0218 | 0.0983 | 3 | 221 |
| #14 | 51.1901 | 0.0696 | 4 | 264 |

## How to Use the Optimized Model

### 1. Load the Optimized Model

```python
import xgboost as xgb

# Load the optimized model
model = xgb.Booster()
model.load_model('model_optimized.ubj')

# Make predictions
import pandas as pd
dmatrix = xgb.DMatrix(X_test)
predictions = model.predict(dmatrix)
```

### 2. Use in ZenML Pipeline

Update `steps/train_model.py` with the best hyperparameters:

```python
model = xgb.XGBRegressor(
    learning_rate=0.046,
    max_depth=3,
    n_estimators=287,
    subsample=0.969,
    colsample_bytree=0.782,
    min_child_weight=4,
    gamma=0.997,
    reg_alpha=2.136,
    reg_lambda=2.286,
    random_state=42,
    objective='reg:squarederror'
)
```

### 3. Run the Optimization Again

To search for even better hyperparameters:

```bash
# In virtual environment
.venv\Scripts\python.exe src\optimize_hyperparameters.py
```

**Note**: Results may vary slightly due to random seed differences, but should be close to 50.71 RMSE.

## MLflow Integration

All optimization runs are logged to MLflow experiment `Turbofan_Optuna_Optimization`.

**View Results**:
```bash
.venv\Scripts\mlflow.exe ui
# Open http://localhost:5000
```

**What's Logged**:
- ✅ All 20 trial parameters
- ✅ RMSE for each trial
- ✅ Best model parameters
- ✅ Final optimized model artifact

## Files Generated

```
model_optimized.ubj          # Optimized XGBoost model (native format)
mlflow.db                    # MLflow tracking database (SQLite)
mlruns/                      # MLflow experiment artifacts
  └── Turbofan_Optuna_Optimization/
      ├── Trial runs (20)
      └── Best model final run
```

## Next Steps

### 1. **Deploy Optimized Model**
Create FastAPI endpoint using `model_optimized.ubj`:

```python
# api/main.py
from fastapi import FastAPI
import xgboost as xgb

model = xgb.Booster()
model.load_model('model_optimized.ubj')

@app.post("/predict")
def predict_rul(features: dict):
    # Convert features to DMatrix
    # Make prediction
    # Return RUL estimate
    pass
```

### 2. **Update ZenML Pipeline**
Integrate best hyperparameters into the production pipeline.

### 3. **Docker Containerization**
Package the optimized model and dependencies:

```dockerfile
# Dockerfile
FROM python:3.12-slim
COPY model_optimized.ubj /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

### 4. **Advanced Optimization (Optional)**
- Increase trials to 50-100 for more thorough search
- Use Optuna's pruning for faster convergence
- Try ensemble methods (stacking, blending)
- Experiment with feature engineering

## Project Status: Optimization Complete ✅

**Deliverables**:
- ✅ Optuna optimization script (`src/optimize_hyperparameters.py`)
- ✅ Improved RMSE: 51.35 → 50.71 cycles
- ✅ Optimized model saved (`model_optimized.ubj`)
- ✅ All runs logged to MLflow
- ✅ Best hyperparameters documented
- ✅ Code committed to GitHub

**Ready for**: FastAPI deployment, Docker containerization, CI/CD integration

---

**Script Execution**:
```bash
cd "D:\cycleing\5eme\R\mlops projet"
.venv\Scripts\python.exe src\optimize_hyperparameters.py
```

**Total Optimization Time**: ~10 seconds (20 trials)  
**Reproducibility**: Set `seed=42` for consistent results
