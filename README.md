# Turbofan RUL Prediction

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
- The `data/` directory is ignored by Git; plan to track datasets with DVC in a next step.

## Next Steps
- Set up a virtual environment and install requirements.
- Initialize DVC and configure a remote to version datasets.
- Set up MLflow tracking and start the first baseline experiment.
- Scaffold a ZenML pipeline and a minimal FastAPI `/predict` endpoint.
- Add Dockerfiles and optional CI.

