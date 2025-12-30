# Turbofan RUL Prediction - MLOps Project

![CI/CD Pipeline](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops/workflows/CI%2FCD%20Pipeline%20-%20Turbofan%20RUL%20MLOps/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Production-grade MLOps pipeline for predicting Remaining Useful Life (RUL) of turbofan engines using NASA CMAPSS dataset.**

---

## 🎯 Project Overview

| Component | Technology | Status |
|-----------|------------|--------|
| Version Control | Git + GitHub | ✅ |
| Data Versioning | DVC | ✅ |
| Experiment Tracking | MLflow | ✅ |
| Pipeline Orchestration | ZenML | ✅ |
| Hyperparameter Optimization | Optuna | ✅ |
| REST API | FastAPI | ✅ |
| Containerization | Docker | ✅ |
| CI/CD | GitHub Actions | ✅ |
| Monitoring | Drift Detection | ✅ |

**Performance:** RMSE = **50.71 cycles** (1.26% improvement over baseline)

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/AymenMB/turbofan-predictive-maintenance-mlops.git
cd turbofan-predictive-maintenance-mlops

# 2. Setup environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data with DVC
dvc pull

# 5. Run API
python -m uvicorn api.main:app --reload --port 8000

# 6. Open Swagger UI → http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Test API
curl http://localhost:8000/health
```

---

## 📚 Documentation

For complete step-by-step implementation details, see **[DOCUMENTATION.md](DOCUMENTATION.md)**

Includes:
- Data preprocessing & RUL calculation
- Model training & optimization
- Pipeline orchestration (ZenML)
- API deployment (FastAPI)
- Docker containerization
- CI/CD automation
- Monitoring & drift detection

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict RUL |
| GET | `/monitoring` | Drift status |
| GET | `/docs` | Swagger UI |

---

## 📁 Project Structure

```
├── api/                    # FastAPI application
├── data/raw/               # NASA CMAPSS dataset (DVC)
├── pipelines/              # ZenML pipeline definitions
├── src/                    # Core ML code
├── steps/                  # ZenML pipeline steps
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker orchestration
├── model_optimized.ubj     # Production model
└── DOCUMENTATION.md        # Complete guide
```

---

## 🔗 Links

- **GitHub:** [turbofan-predictive-maintenance-mlops](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops)
- **API Docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000

---

**Author:** Aymen Mabrouk  
**Institution:** Ecole Polytechnique Sousse  
**Version:** 1.1.0

