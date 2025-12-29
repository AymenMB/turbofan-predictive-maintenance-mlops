# Turbofan RUL Prediction

![CI/CD Pipeline](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops/workflows/CI%2FCD%20Pipeline%20-%20Turbofan%20RUL%20MLOps/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Production-grade MLOps project using NASA CMAPSS turbofan engine data for Remaining Useful Life (RUL) prediction.

## Repository Structure

```
.
├── api/                # FastAPI application (inference service)
├── data/
│   ├── raw/            # Raw CMAPSS files (train_FD*, test_FD*, RUL_FD*)
│   └── processed/      # Cleaned/feature-engineered datasets (generated)
├── docker/             # Dockerfiles and container configs
├── notebooks/          # EDA and experimentation notebooks
├── pipelines/          # ZenML pipelines (data -> train -> eval -> export)
├── src/                # Reusable source code (preprocessing, training, eval)
├── .gitignore          # Ignored files (data/ is ignored; DVC will manage later)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Data
- The dataset files provided in this repository were moved to `data/raw/`.
- The `data/` directory is ignored by Git and managed by DVC.

## Data Versioning (DVC)

This project uses **DVC (Data Version Control)** to version the dataset and ensure reproducibility.

### Initial Setup (Already Done)
```bash
# DVC is initialized and data/raw is tracked
dvc init
dvc add data/raw
dvc remote add -d local_storage D:\dvc_store
dvc push
```

### Clone & Pull Data
If you clone this repository, the actual data files are **not included** in Git. To retrieve them:

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# Pull the data from DVC remote
dvc pull
```

This will download all data files from the configured remote storage into `data/raw/`.

### Verify Data
After pulling, verify the data is available:
```bash
ls data/raw/  # Should show train_FD*.txt, test_FD*.txt, RUL_FD*.txt
```

### Updating Data
If you modify or add new data:
```bash
dvc add data/raw
dvc push
git add data/raw.dvc data/.gitignore
git commit -m "Update dataset"
```

## Next Steps
- Set up a virtual environment and install requirements.
- Initialize DVC and configure a remote to version datasets.
- Set up MLflow tracking and start the first baseline experiment.
- Scaffold a ZenML pipeline and a minimal FastAPI `/predict` endpoint.
- Add Dockerfiles and optional CI.

