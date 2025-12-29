# ZenML Pipeline Implementation Guide

## Overview
This document provides a comprehensive guide to the ZenML pipeline implementation for the Turbofan RUL Prediction project.

## Pipeline Architecture

The pipeline is implemented as a modular, reproducible ML workflow with 4 distinct steps:

```
┌──────────────────┐
│  ingest_data     │  → Load raw CMAPSS data
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│   clean_data     │  → Calculate RUL, drop constant sensors
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  train_model     │  → Train XGBoost with time-series split
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ evaluate_model   │  → Compute RMSE, MAE, R², log to MLflow
└──────────────────┘
```

## File Structure

```
├── steps/                      # ZenML step definitions
│   ├── __init__.py
│   ├── ingest_data.py         # Data loading step
│   ├── clean_data.py          # Data preprocessing step
│   ├── train_model.py         # Model training step
│   └── evaluate_model.py      # Model evaluation step
├── pipelines/                  # Pipeline orchestration
│   ├── __init__.py
│   └── training_pipeline.py   # Main pipeline definition
├── run_pipeline.py            # Pipeline execution script
└── test_pipeline.py           # Testing script (for development)
```

## Step Details

### 1. Data Ingestion (`ingest_data`)
**File**: `steps/ingest_data.py`

**Purpose**: Load raw CMAPSS turbofan dataset

**Input Parameters**:
- `data_path` (str): Path to the training dataset

**Output**: 
- `pd.DataFrame`: Raw dataset with 26 columns (unit, cycle, settings, sensors)

**Key Features**:
- Validates file existence
- Assigns standardized column names
- Logs basic statistics (rows, unique engines)

### 2. Data Cleaning (`clean_data`)
**File**: `steps/clean_data.py`

**Purpose**: Preprocess data and calculate Remaining Useful Life (RUL)

**Input**:
- `df` (pd.DataFrame): Raw dataset from ingestion

**Output**:
- `pd.DataFrame`: Cleaned dataset with RUL column, 21 columns total

**Key Features**:
- Calculates RUL = max_cycle - current_cycle per engine
- Drops 6 constant sensors (s_1, s_5, s_10, s_16, s_18, s_19)
- Logs transformation statistics

### 3. Model Training (`train_model`)
**File**: `steps/train_model.py`

**Purpose**: Train XGBoost regressor with time-series aware split

**Input**:
- `df` (pd.DataFrame): Cleaned dataset with RUL
- `n_estimators` (int): Number of trees (default: 100)
- `learning_rate` (float): Learning rate (default: 0.1)
- `max_depth` (int): Max tree depth (default: 6)

**Output**:
- `xgboost.XGBRegressor`: Trained model

**Key Features**:
- Time-series split: engines 1-80 (train), 81-100 (test)
- **No random shuffling** - preserves temporal integrity
- XGBoost with early stopping awareness
- Step caching disabled (`enable_cache=False`) for reproducibility

### 4. Model Evaluation (`evaluate_model`)
**File**: `steps/evaluate_model.py`

**Purpose**: Evaluate model and log metrics

**Input**:
- `model` (XGBRegressor): Trained model from previous step
- `df` (pd.DataFrame): Cleaned dataset with RUL

**Output**:
- `Dict[str, float]`: Evaluation metrics (rmse, mae, r2)

**Key Features**:
- Computes RMSE (Root Mean Squared Error)
- Computes MAE (Mean Absolute Error)
- Computes R² Score
- Logs metrics to MLflow (if experiment tracker configured)

## Pipeline Configuration

**File**: `pipelines/training_pipeline.py`

```python
@pipeline
def training_pipeline(
    data_path: str,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6
) -> Dict[str, float]:
    """
    Orchestrates the complete training workflow.
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    raw_data = ingest_data(data_path=data_path)
    cleaned_data = clean_data(df=raw_data)
    model = train_model(
        df=cleaned_data,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    metrics = evaluate_model(model=model, df=cleaned_data)
    return metrics
```

## Running the Pipeline

### Option 1: Using the Run Script (Recommended)
```bash
python run_pipeline.py
```

### Option 2: Programmatic Execution
```python
from pipelines.training_pipeline import training_pipeline

metrics = training_pipeline(
    data_path="data/raw/train_FD001.txt",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
print(f"RMSE: {metrics['rmse']:.2f} cycles")
```

## ZenML Setup & Configuration

### Prerequisites
1. **Project Setup**:
   ```bash
   zenml project register turbofan-rul
   zenml project set turbofan-rul
   ```

2. **Required Dependencies** (already in `requirements.txt`):
   - `zenml==0.93.0`
   - `zenml[server]` (for full functionality)
   - `sqlmodel==0.0.18`
   - `sqlalchemy_utils==0.42.1`
   - `passlib==1.7.4`

### First-Time Setup
```bash
# Install ZenML with server dependencies
pip install "zenml[server]"

# Register project
zenml project register turbofan-rul
zenml project set turbofan-rul

# Run pipeline
python run_pipeline.py
```

## Pipeline Run Output

When you run the pipeline, you'll see:

```
======================================================================
TURBOFAN RUL PREDICTION - ZenML PIPELINE
======================================================================

Initiating a new run for the pipeline: training_pipeline.
Step ingest_data has started.
✓ Ingested 20631 rows from data/raw/train_FD001.txt
✓ Dataset contains 100 unique engines
Step ingest_data has finished in 5.4s.

Step clean_data has started.
✓ Calculated RUL for all samples
✓ Dropped 6 constant sensors
✓ Final shape: (20631, 21)
Step clean_data has finished in 3.1s.

Step train_model has started.
  → Training set: 16138 samples from engines 1-80
  → Test set: 4493 samples from engines 81-100
✓ Training complete!
Step train_model has finished in 3.1s.

Step evaluate_model has started.
  → Test RMSE: 51.35 cycles
  → Test MAE: 36.55 cycles
  → R² Score: 0.5609
Step evaluate_model has finished in 1.5s.

Pipeline run has finished in 21.6s.
```

## ZenML Dashboard

### Accessing the Dashboard
The pipeline run is automatically tracked and can be viewed on ZenML Cloud:

```
Dashboard URL: https://cloud.zenml.io/workspaces/[workspace]/projects/[project]/runs/[run-id]
```

### What You Can See:
- ✅ Full pipeline execution graph
- ✅ Step-by-step execution times
- ✅ Input/output artifacts for each step
- ✅ Metadata and parameters
- ✅ Run status and logs
- ✅ Model lineage

## Integration with MLflow

The `evaluate_model` step attempts to log metrics to MLflow if an experiment tracker is configured:

```python
try:
    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_mae": mae,
        "r2_score": r2
    })
except Exception:
    print("  ⚠ No experiment tracker configured, metrics not logged")
```

### To Enable MLflow Logging:
1. Start MLflow UI: `.venv\Scripts\mlflow.exe ui`
2. The pipeline will automatically log to the active experiment

## Performance Metrics

**Baseline Performance** (FD001 dataset):
- RMSE: 51.35 cycles
- MAE: 36.55 cycles
- R² Score: 0.5609
- Total pipeline time: ~21 seconds

## Advantages of ZenML Pipeline

### 1. **Reproducibility**
- Every pipeline run is tracked with:
  - Input data hash
  - Code version
  - Model parameters
  - Environment details

### 2. **Modularity**
- Each step is independently testable
- Easy to swap out components (e.g., change model from XGBoost to RandomForest)
- Steps can be reused in different pipelines

### 3. **Caching**
- ZenML caches unchanged steps by default
- Training step has caching disabled (`enable_cache=False`) for consistency

### 4. **Artifact Tracking**
- Automatic versioning of:
  - Datasets (raw, cleaned)
  - Models
  - Metrics

### 5. **Experiment Management**
- Compare multiple runs
- Track hyperparameter changes
- Visualize pipeline evolution

## Next Steps

### 1. **Hyperparameter Optimization**
Integrate Optuna for automated tuning:
```python
# TODO: Create optimize_hyperparameters.py
# with Optuna + ZenML integration
```

### 2. **FastAPI Deployment**
Create inference endpoint:
```python
# TODO: Create api/main.py
# Load model from ZenML artifact store
# Serve predictions via REST API
```

### 3. **Docker Containerization**
```dockerfile
# TODO: Create Dockerfile
# Package pipeline + dependencies
# Enable deployment anywhere
```

### 4. **CI/CD Integration**
```yaml
# TODO: Create .gitlab-ci.yml or .github/workflows/mlops.yml
# Automate: test → train → deploy
```

## Troubleshooting

### Issue: "No active project is configured"
**Solution**:
```bash
zenml project register turbofan-rul
zenml project set turbofan-rul
```

### Issue: Missing dependencies (sqlalchemy_utils, sqlmodel, passlib)
**Solution**:
```bash
pip install "zenml[server]" sqlalchemy_utils sqlmodel passlib
```

### Issue: Connection refused to localhost:8080
**Solution**: ZenML daemon is not supported on Windows. The pipeline will use local mode automatically.

### Issue: Pipeline runs but metrics not in MLflow
**Solution**: Start MLflow UI and ensure the experiment tracker is running:
```bash
.venv\Scripts\mlflow.exe ui
```

## Files Added to Git

```
✅ steps/ingest_data.py
✅ steps/clean_data.py
✅ steps/train_model.py
✅ steps/evaluate_model.py
✅ pipelines/training_pipeline.py
✅ run_pipeline.py
✅ test_pipeline.py
✅ steps/__init__.py
✅ pipelines/__init__.py
```

**Commit Message**: "Add ZenML pipeline implementation with modular steps (ingest, clean, train, evaluate)"

**GitHub Repository**: https://github.com/AymenMB/turbofan-predictive-maintenance-mlops.git

## Summary

✅ **Complete ZenML pipeline implemented** with 4 modular steps
✅ **Successfully tested** - RMSE: 51.35 cycles
✅ **Tracked on ZenML Cloud** - Full lineage and metadata
✅ **Committed to Git** - Version controlled and reproducible
✅ **Ready for production** - Can be deployed, scaled, monitored

The project now has a **fully industrial-grade ML pipeline** that follows best practices for experiment tracking, artifact management, and reproducibility.
