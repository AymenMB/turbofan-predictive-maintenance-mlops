# 📚 Turbofan RUL MLOps - Complete Documentation

> **Production-grade MLOps pipeline for predicting Remaining Useful Life (RUL) of turbofan engines using NASA CMAPSS dataset.**

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [Project Structure](#3-project-structure)
4. [Step-by-Step Implementation](#4-step-by-step-implementation)
   - [4.1 Git & Version Control](#41-git--version-control)
   - [4.2 Data Version Control (DVC)](#42-data-version-control-dvc)
   - [4.3 Data Preprocessing](#43-data-preprocessing)
   - [4.4 Model Training](#44-model-training-baseline)
   - [4.5 Experiment Tracking (MLflow)](#45-experiment-tracking-mlflow)
   - [4.6 Pipeline Orchestration (ZenML)](#46-pipeline-orchestration-zenml)
   - [4.7 Hyperparameter Optimization (Optuna)](#47-hyperparameter-optimization-optuna)
   - [4.8 REST API Deployment (FastAPI)](#48-rest-api-deployment-fastapi)
   - [4.9 Docker Containerization](#49-docker-containerization)
   - [4.10 CI/CD Pipeline (GitHub Actions)](#410-cicd-pipeline-github-actions)
   - [4.11 Monitoring & Drift Detection (Bonus)](#411-monitoring--drift-detection-bonus)
5. [Performance Results](#5-performance-results)
6. [How to Run](#6-how-to-run)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Overview

### 🎯 Objective

Build a complete **end-to-end MLOps workflow** for predictive maintenance of turbofan engines:

| Component | Technology |
|-----------|------------|
| Version Control | Git + GitHub |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | ZenML |
| Hyperparameter Optimization | Optuna |
| Model Serving | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Custom Drift Detection |

### 📊 Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

| File | Description | Records |
|------|-------------|---------|
| `train_FD001.txt` | Training data (100 engines) | 20,631 |
| `test_FD001.txt` | Test data | 13,096 |
| `RUL_FD001.txt` | Ground truth RUL | 100 |

**Features:**
- 3 operational settings (`setting_1`, `setting_2`, `setting_3`)
- 21 sensor measurements (`s_1` to `s_21`)
- Target: Remaining Useful Life (RUL) in cycles

### 🏆 Final Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **RMSE** | 51.35 cycles | **50.71 cycles** | **1.26%** ⬇️ |
| MAE | 36.55 cycles | - | - |
| R² | 0.5609 | - | - |

---

## 2. Quick Start

```bash
# 1. Clone repository
git clone https://github.com/AymenMB/turbofan-predictive-maintenance-mlops.git
cd turbofan-predictive-maintenance-mlops

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data with DVC
dvc pull

# 5. Run API
python -m uvicorn api.main:app --reload --port 8000

# 6. Open Swagger UI
# http://localhost:8000/docs
```

---

## 3. Project Structure

```
turbofan-predictive-maintenance-mlops/
│
├── 📁 .github/workflows/         # CI/CD pipeline
│   └── ci_cd.yaml               # GitHub Actions workflow
│
├── 📁 api/                       # FastAPI application
│   ├── __init__.py
│   └── main.py                  # API endpoints (v1.1.0)
│
├── 📁 data/
│   └── raw/                     # NASA CMAPSS data (DVC tracked)
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── 📁 pipelines/                 # ZenML pipeline definitions
│   └── training_pipeline.py
│
├── 📁 src/                       # Core ML code
│   ├── data_preprocessing.py    # Data loading & RUL calculation
│   ├── train_model.py           # XGBoost baseline training
│   └── optimize_hyperparameters.py  # Optuna optimization
│
├── 📁 steps/                     # ZenML pipeline steps
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── 📄 Dockerfile                 # Container definition
├── 📄 docker-compose.yml         # Docker orchestration
├── 📄 requirements.txt           # Python dependencies
├── 📄 model_optimized.ubj        # Production model (RMSE: 50.71)
├── 📄 simulate_drift.py          # Drift detection demo
└── 📄 README.md                  # Project overview
```

---

## 4. Step-by-Step Implementation

---

### 4.1 Git & Version Control

**Objective:** Initialize Git repository with clean structure.

```bash
# Initialize repository
git init
git remote add origin https://github.com/AymenMB/turbofan-predictive-maintenance-mlops.git

# Create .gitignore
# Ignores: __pycache__, .venv, mlruns/, data/, *.ubj, etc.

# Commit and push
git add .
git commit -m "Initial project structure"
git push -u origin main
```

**Repository:** [github.com/AymenMB/turbofan-predictive-maintenance-mlops](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops)

---

### 4.2 Data Version Control (DVC)

**Objective:** Track large dataset files outside Git.

```bash
# Initialize DVC
dvc init

# Track raw data
dvc add data/raw

# Configure local remote storage
dvc remote add -d local_storage D:\dvc_store

# Push data to remote
dvc push

# Commit DVC files to Git
git add data/raw.dvc data/.gitignore .dvc/
git commit -m "Track data with DVC"
```

**Reproducibility:** Anyone can run `dvc pull` to get the exact same dataset.

---

### 4.3 Data Preprocessing

**Objective:** Load data, calculate RUL, drop constant sensors.

**File:** `src/data_preprocessing.py`

```python
def load_and_process_data(file_path):
    """
    Load CMAPSS data and calculate Remaining Useful Life.
    
    Steps:
    1. Load whitespace-separated file
    2. Assign column names (unit, cycle, 3 settings, 21 sensors)
    3. Calculate RUL = max_cycle - current_cycle per engine
    4. Drop 6 constant sensors: s_1, s_5, s_10, s_16, s_18, s_19
    
    Returns:
        DataFrame with 21 columns (3 settings + 15 sensors + RUL)
    """
```

**RUL Calculation:**
```python
# For each engine, RUL decreases from max to 0
max_cycles = df.groupby('unit')['cycle'].max()
df['RUL'] = df.apply(lambda x: max_cycles[x['unit']] - x['cycle'], axis=1)
```

**Output:** 20,631 samples × 21 features

---

### 4.4 Model Training (Baseline)

**Objective:** Train XGBoost regressor with time-series split.

**File:** `src/train_model.py`

```python
# Time-series split (NO random shuffling)
train_engines = range(1, 81)   # Engines 1-80
test_engines = range(81, 101)  # Engines 81-100

# XGBoost Configuration
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
```

**Baseline Performance:**
- **RMSE:** 51.35 cycles
- **MAE:** 36.55 cycles
- **R²:** 0.5609

---

### 4.5 Experiment Tracking (MLflow)

**Objective:** Log experiments, parameters, metrics, and artifacts.

```python
import mlflow

# Set experiment
mlflow.set_experiment("Turbofan_RUL_Prediction")

with mlflow.start_run(run_name="baseline_xgboost"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 6)
    
    # Log metrics
    mlflow.log_metric("rmse", 51.35)
    mlflow.log_metric("mae", 36.55)
    mlflow.log_metric("r2", 0.5609)
    
    # Log model artifact
    mlflow.xgboost.log_model(model, "model")
```

**View MLflow UI:**
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

### 4.6 Pipeline Orchestration (ZenML)

**Objective:** Create reproducible ML pipeline.

**Pipeline Architecture:**
```
ingest_data → clean_data → train_model → evaluate_model
```

**Files:**
- `steps/ingest_data.py` - Load raw data
- `steps/clean_data.py` - Calculate RUL, drop sensors
- `steps/train_model.py` - Train XGBoost
- `steps/evaluate_model.py` - Compute RMSE, MAE, R²
- `pipelines/training_pipeline.py` - Orchestrate steps

**Run Pipeline:**
```bash
python run_pipeline.py
```

**ZenML Cloud:** Connected to ZenML Cloud for visualization.

---

### 4.7 Hyperparameter Optimization (Optuna)

**Objective:** Find optimal hyperparameters using TPE sampler.

**File:** `src/optimize_hyperparameters.py`

**Search Space:**
| Parameter | Range |
|-----------|-------|
| learning_rate | 0.01 - 0.3 |
| max_depth | 3 - 10 |
| n_estimators | 50 - 300 |
| subsample | 0.6 - 1.0 |
| colsample_bytree | 0.6 - 1.0 |
| min_child_weight | 1 - 10 |
| gamma | 0.0 - 5.0 |
| reg_alpha | 0.0 - 5.0 |
| reg_lambda | 0.0 - 5.0 |

**Best Hyperparameters (Trial #11):**
```python
{
    'learning_rate': 0.046,
    'max_depth': 3,
    'n_estimators': 287,
    'subsample': 0.969,
    'colsample_bytree': 0.782,
    'min_child_weight': 4,
    'gamma': 0.997,
    'reg_alpha': 2.136,
    'reg_lambda': 2.286
}
```

**Result:** RMSE improved from 51.35 → **50.71 cycles** (1.26% better)

---

### 4.8 REST API Deployment (FastAPI)

**Objective:** Serve model predictions via HTTP.

**File:** `api/main.py`

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Predict RUL |
| GET | `/model-info` | Model details |
| GET | `/monitoring` | Drift status |
| GET | `/monitoring/reset` | Clear buffer |

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "s_2": 641.82,
    "s_3": 1589.70,
    ...
  }'
```

**Response:**
```json
{
  "RUL": 112.45,
  "status": "Healthy",
  "confidence": "High"
}
```

**Status Classification:**
- `Critical`: RUL < 30 (immediate action)
- `Warning`: RUL < 80 (schedule maintenance)
- `Healthy`: RUL ≥ 80 (normal operation)

---

### 4.9 Docker Containerization

**Objective:** Package application in container.

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Create non-root user (security)
RUN useradd --create-home appuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api/ api/
COPY model_optimized.ubj .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
# Build image
docker build -t turbofan-rul-api:latest .

# Run container
docker run -d -p 8000:8000 --name turbofan-api turbofan-rul-api:latest

# Using docker-compose
docker-compose up -d
```

---

### 4.10 CI/CD Pipeline (GitHub Actions)

**Objective:** Automate testing, building, and validation.

**File:** `.github/workflows/ci_cd.yaml`

**Pipeline Jobs:**

```
┌─────────────┐
│ Test & Lint │  ← flake8, pytest, black
└──────┬──────┘
       │
       ├─────────────┬─────────────┐
       │             │             │
       ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌──────────┐
│ Build Docker │ │ ML Test  │ │ Security │
└──────────────┘ └──────────┘ └──────────┘
       │             │             │
       └──────┬──────┴─────────────┘
              ▼
     ┌─────────────────┐
     │ Deploy Summary  │
     └─────────────────┘
```

**Triggers:** Push/PR to `main` branch

**Badge:** ![CI/CD](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops/workflows/CI%2FCD%20Pipeline%20-%20Turbofan%20RUL%20MLOps/badge.svg)

---

### 4.11 Monitoring & Drift Detection (Bonus)

**Objective:** Detect when input data deviates from training distribution.

**Implementation:**
- Store last 100 predictions in memory (circular buffer)
- Compare recent means vs training baseline
- Flag features with >20% deviation

**Endpoints:**
```bash
# Check drift status
GET /monitoring

# Reset monitoring buffer
GET /monitoring/reset
```

**Simulation:**
```bash
python simulate_drift.py
```

**Phase 1 (Normal Data):** No drift detected  
**Phase 2 (Corrupted Data):** 17/21 features flagged with ~25% deviation

---

## 5. Performance Results

### Model Comparison

| Model | RMSE (cycles) | MAE | R² |
|-------|---------------|-----|-----|
| Baseline XGBoost | 51.35 | 36.55 | 0.5609 |
| **Optimized XGBoost** | **50.71** | - | - |

### API Performance

| Metric | Value |
|--------|-------|
| Startup Time | ~3 sec |
| Prediction Latency | <200ms |
| Health Check | <100ms |
| Docker Image Size | ~500MB |

### CI/CD Pipeline

| Job | Duration |
|-----|----------|
| Test & Lint | ~2-3 min |
| Build Docker | ~3-5 min |
| ML Validation | ~1-2 min |
| Security Scan | ~2-3 min |
| **Total** | **~8-13 min** |

---

## 6. How to Run

### Option 1: Local Development

```bash
# Setup
cd "d:\cycleing\5eme\R\mlops projet"
.\.venv\Scripts\activate

# Run API
python -m uvicorn api.main:app --reload --port 8000

# Run optimization
python src/optimize_hyperparameters.py

# Run pipeline
python run_pipeline.py
```

### Option 2: Docker

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 3: Full MLflow + ZenML Stack

```bash
# Terminal 1: MLflow
mlflow ui --port 5000

# Terminal 2: API
python -m uvicorn api.main:app --port 8000

# Terminal 3: Pipeline
python run_pipeline.py
```

---

## 7. API Reference

### POST /predict

**Request Body:**
```json
{
  "setting_1": -0.0007,
  "setting_2": -0.0004,
  "setting_3": 100.0,
  "s_1": 518.67,
  "s_2": 641.82,
  "s_3": 1589.70,
  "s_4": 1400.60,
  "s_5": 14.62,
  "s_6": 21.61,
  "s_7": 554.36,
  "s_8": 2388.06,
  "s_9": 9046.19,
  "s_10": 1.30,
  "s_11": 47.47,
  "s_12": 521.66,
  "s_13": 2388.02,
  "s_14": 8138.62,
  "s_15": 8.4195,
  "s_16": 0.03,
  "s_17": 392,
  "s_18": 2388,
  "s_19": 100.0,
  "s_20": 39.06,
  "s_21": 23.4190
}
```

**Response:**
```json
{
  "RUL": 112.45,
  "status": "Healthy",
  "confidence": "High"
}
```

### GET /health

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "model_optimized.ubj"
}
```

### GET /monitoring

```json
{
  "drift_detected": false,
  "status": "No significant drift detected",
  "metrics": {
    "max_deviation_pct": 5.2,
    "threshold_pct": 20.0,
    "drifted_features": []
  },
  "recent_requests": 50
}
```

---

## 8. Troubleshooting

### DVC Pull Fails

```bash
# Check remote configuration
dvc remote list

# Manual pull
dvc pull -v
```

### API Won't Start

```bash
# Check if port is in use
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F

# Try different port
python -m uvicorn api.main:app --port 8001
```

### Docker Build Fails

```bash
# Clear cache
docker builder prune

# Rebuild
docker build --no-cache -t turbofan-rul-api:latest .
```

### Model Not Loading

```bash
# Check file exists
ls model_optimized.ubj

# Verify XGBoost version
pip show xgboost
```

---

## 📝 Deliverables Checklist

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Git repository with clean structure | ✅ |
| 2 | DVC for data versioning | ✅ |
| 3 | Data preprocessing with RUL calculation | ✅ |
| 4 | Baseline model training | ✅ |
| 5 | MLflow experiment tracking | ✅ |
| 6 | ZenML pipeline orchestration | ✅ |
| 7 | Optuna hyperparameter optimization | ✅ |
| 8 | FastAPI REST deployment | ✅ |
| 9 | Docker containerization | ✅ |
| 10 | CI/CD with GitHub Actions | ✅ |
| 11 | Comprehensive testing | ✅ |
| 12 | Documentation | ✅ |
| **Bonus** | Monitoring & drift detection | ✅ |

---

## 🔗 Links

- **GitHub:** [turbofan-predictive-maintenance-mlops](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops)
- **API Docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000

---

**Author:** Aymen Mabrouk  
**Institution:** Ecole Polytechnique Sousse  
**Version:** 1.1.0  
**Date:** December 2025  

---

🚀 **Production-Ready MLOps Pipeline Complete!**
