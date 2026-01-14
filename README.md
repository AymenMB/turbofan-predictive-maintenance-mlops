# 🛩️ AeroGuard AI - Mini-projet MLOps

<div align="center">

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![Azure](https://img.shields.io/badge/azure-deployed-0078D4.svg)
![MLflow](https://img.shields.io/badge/mlflow-tracking-orange.svg)
![ZenML](https://img.shields.io/badge/zenml-pipeline-purple.svg)

**Prédiction de la Durée de Vie Restante (RUL) des Moteurs Turbofan**

*Un workflow MLOps complet de bout en bout*

</div>

---

## 📋 Informations du Projet

| | |
|---|---|
| **Étudiant** | Aymen MABROUK |
| **Encadrant** | Dr. Salah GONTARA |
| **Institution** | École Polytechnique Sousse |
| **Module** | MLOps |
| **Année** | 2025-2026 |

---

## 🎯 Objectif du Projet

Ce mini-projet MLOps implémente un **workflow complet de bout en bout** pour la maintenance prédictive des moteurs turbofan, incluant :

- ✅ Gestion du code (Git)
- ✅ Conteneurisation (Docker / Docker Compose)
- ✅ Versioning des données (DVC)
- ✅ Suivi d'expériences (MLflow)
- ✅ Pipeline ML (ZenML)
- ✅ Optimisation (Optuna)
- ✅ CI/CD (GitHub Actions)
- ✅ Déploiement (API FastAPI sur Azure)
- ✅ **Bonus : Monitoring** (détection de drift)
- ✅ **Bonus : Retrain automatique**

---

## 📊 Cas d'Usage & Dataset

### Dataset : NASA C-MAPSS (FD001)

| Caractéristique | Valeur |
|-----------------|--------|
| **Source** | NASA Prognostics Center |
| **Type** | Série temporelle / Régression |
| **Taille** | 100 moteurs, ~21,000 cycles |
| **Features** | 21 capteurs + 3 paramètres opérationnels |
| **Target** | RUL (Remaining Useful Life) |

### Modèle : XGBoost avec Feature Engineering

| Métrique | Valeur |
|----------|--------|
| **RMSE** | **18.64 cycles** |
| **R²** | **0.79** |
| **Amélioration** | 63% vs baseline |

---

## 📁 Structure du Projet

```
📦 turbofan-predictive-maintenance-mlops
├── 📂 api/                    # FastAPI application
│   ├── main.py               # Endpoints API
│   └── __init__.py
├── 📂 data/                   # Dataset (DVC)
│   ├── raw/                  # Données brutes
│   └── raw.dvc               # Fichier DVC tracking
├── 📂 pipelines/              # ZenML pipelines
│   └── training_pipeline.py  # Pipeline d'entraînement
├── 📂 steps/                  # ZenML steps
│   ├── ingest_data.py        # Ingestion données
│   ├── clean_data.py         # Prétraitement
│   ├── train_model.py        # Entraînement
│   └── evaluate_model.py     # Évaluation
├── 📂 src/                    # Code ML
│   ├── data_preprocessing.py # Feature engineering
│   ├── optimize_hyperparameters.py # Optuna
│   └── train.py              # Script training
├── 📂 .github/workflows/      # CI/CD
│   ├── ci_cd.yaml            # Pipeline CI
│   └── deploy-azure.yaml     # Déploiement Azure
├── 📂 mlruns/                 # MLflow experiments
├── 📂 screenshots/            # Captures d'écran
├── 📄 Dockerfile              # Image Docker
├── 📄 docker-compose.yml      # Orchestration
├── 📄 retrain.py              # Script retrain auto
├── 📄 simulate_drift.py       # Simulation drift
└── 📄 README.md               # Ce fichier
```

---

## 🔧 3.2 Gestion du Code (Git)

### Branches
```bash
$ git branch -a
* main                    # Production
  dev                     # Développement
  remotes/origin/main
  remotes/origin/dev
```

### Tags (Versioning)
```bash
$ git tag -l
v1    # Version initiale
v2    # Améliorations
v3    # Version finale avec bonus
```

### Repository GitHub
🔗 https://github.com/AymenMB/turbofan-predictive-maintenance-mlops

---

## 🐳 3.3 Conteneurisation (Docker)

### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements-api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ ./api/
COPY model_optimized.ubj .
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import requests; requests.get('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - '8000:8000'
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
```

### Commandes
```bash
# Lancer le conteneur
docker-compose up -d

# Vérifier le status
docker-compose ps
# Résultat: turbofan-rul-api   Up (healthy)   0.0.0.0:8000->8000/tcp
```

---

## 📦 3.4 Versioning des Données (DVC)

### 🎯 Ce que nous avons fait
DVC (Data Version Control) permet de versionner les gros fichiers de données sans les mettre dans Git.

**Actions réalisées:**
1. Initialisé DVC avec `dvc init`
2. Tracké le dossier `data/raw/` contenant le dataset NASA
3. Configuré un remote de stockage pour sauvegarder les données
4. Créé le fichier `data/raw.dvc` qui référence les données

### Configuration
```bash
$ dvc remote list
local_storage   D:\dvc_store    (default)
```

### Fichiers trackés
```
data/raw.dvc
├── 12 fichiers (44.9 MB total)
├── train_FD001.txt   # Données d'entraînement (100 moteurs)
├── test_FD001.txt    # Données de test
└── RUL_FD001.txt     # Labels RUL pour le test
```

### 🔍 Comment vérifier
```bash
# Vérifier que les données sont synchronisées
$ dvc status
Data and pipelines are up to date.   ✅ Signifie que tout est OK!

# Récupérer les données (pour un nouveau clone)
$ dvc pull
# Télécharge les 12 fichiers depuis le remote

# Sauvegarder les données modifiées
$ dvc push
# Envoie les données vers le remote
```

### 📁 Fichier data/raw.dvc (contenu)
```yaml
outs:
- md5: 4f031cda497f36cac6922c0e7238b1f9.dir
  size: 44913306
  nfiles: 12
  hash: md5
  path: raw
```

---

## 📈 3.5 Experiment Tracking (MLflow)

### 🎯 Ce que nous avons fait
MLflow permet de tracker toutes les expériences ML: paramètres, métriques et modèles.

**Actions réalisées:**
1. Intégré MLflow dans les scripts d'entraînement
2. Créé des expériences pour organiser les runs
3. Loggé les hyperparamètres de chaque run
4. Loggé les métriques (RMSE, MAE, R²)
5. Sauvegardé les modèles comme artefacts

### 📂 Structure des fichiers MLflow
```
mlruns/
├── 1/                          # Experiment 1: Turbofan_RUL_Prediction
│   ├── 5bf6e15b.../           # Run 1
│   │   ├── artifacts/         # Modèles sauvegardés
│   │   ├── metrics/           # RMSE, MAE, R²
│   │   └── params/            # Hyperparamètres
│   ├── 6371496a.../           # Run 2
│   ├── 99283140.../           # Run 3
│   └── d013f742.../           # Run 4
└── 2/                          # Experiment 2: Optuna
```

### Runs enregistrés (4+ runs)
| Run | RMSE | R² | Description |
|-----|------|-----|-------------|
| Baseline | 50.71 | 0.56 | Sans feature engineering |
| Feature Engineering | 18.89 | 0.78 | Rolling windows + normalization |
| Optuna Optimized | **18.64** | **0.79** | Meilleur run, hyperparamètres optimaux |
| Variations | ~19-22 | 0.75+ | Tests avec différents paramètres |

### Code d'intégration (extrait de train_model.py)
```python
import mlflow

# Configurer l'expérience
mlflow.set_experiment("Turbofan_RUL_Prediction")

# Logger les paramètres
mlflow.log_param("n_estimators", 300)
mlflow.log_param("learning_rate", 0.05)

# Logger les métriques
mlflow.log_metric("rmse", 18.64)
mlflow.log_metric("r2", 0.79)

# Sauvegarder le modèle
mlflow.log_artifact("model_optimized.ubj")
```

### 🔍 Comment vérifier
```bash
# Lancer l'interface MLflow
mlflow ui --port 5000

# Ouvrir dans le navigateur
http://localhost:5000

# Vous verrez:
# - Liste des experiments
# - Tous les runs avec leurs métriques
# - Graphiques de comparaison
# - Artefacts téléchargeables
```

### Artefacts loggés pour chaque run
- ✅ **Paramètres**: learning_rate, max_depth, n_estimators, subsample, etc.
- ✅ **Métriques**: RMSE, MAE, R², durée d'entraînement
- ✅ **Artefacts**: model_optimized.ubj, feature_columns.txt

---

## 🔄 3.6 Pipeline MLOps (ZenML)

### 🎯 Ce que nous avons fait
ZenML orchestre le pipeline ML en étapes modulaires et reproductibles.

**Actions réalisées:**
1. Créé 4 steps réutilisables dans `steps/`
2. Assemblé les steps dans `pipelines/training_pipeline.py`
3. Chaque step a ses inputs/outputs typés
4. Intégration avec MLflow pour le tracking

### Architecture du Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ingest_data │────▶│  clean_data  │────▶│  train_model │────▶│evaluate_model│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     Load            Feature               XGBoost             RMSE, MAE,
    FD001.txt        Engineering          Training               R²
```

### 📂 Fichiers du pipeline

#### 1. `pipelines/training_pipeline.py` (58 lignes)
```python
from zenml import pipeline
from steps import ingest_data, clean_data, train_model, evaluate_model

@pipeline
def training_pipeline(data_path: str = "data/raw/train_FD001.txt"):
    raw_data = ingest_data(data_path=data_path)      # Step 1
    cleaned_data = clean_data(df=raw_data)           # Step 2
    model = train_model(df=cleaned_data)             # Step 3
    metrics = evaluate_model(model=model, df=cleaned_data)  # Step 4
    return metrics
```

#### 2. `steps/ingest_data.py` - Chargement des données
- Lit le fichier `train_FD001.txt`
- Parse les 24 colonnes (unit, cycle, settings, sensors)
- Retourne un DataFrame pandas

#### 3. `steps/clean_data.py` - Feature Engineering  
- Calcule le RUL pour chaque engine
- Applique le RUL clipping à 125 cycles
- Crée les rolling features (mean, std sur 5 cycles)
- Normalise les capteurs

#### 4. `steps/train_model.py` (118 lignes) - Entraînement
- Split time-series aware (engines 1-80 train, 81-100 test)
- Entraîne XGBoost avec les hyperparamètres optimaux
- Intègre MLflow pour le logging

#### 5. `steps/evaluate_model.py` - Évaluation
- Calcule RMSE, MAE, R²
- Log les métriques dans MLflow
- Affiche le rapport de performance

### 🔍 Comment exécuter
```bash
# Exécuter le pipeline
python run_pipeline.py

# Output attendu:
# ======================================================================
# TURBOFAN RUL PREDICTION - ZenML PIPELINE
# ======================================================================
# Initiating a new run for the pipeline: training_pipeline.
# Step ingest_data has started.
# Step clean_data has started.
# Step train_model has started.
# Step evaluate_model has started.
# Pipeline run completed successfully!
```

---

## ⚙️ 3.7 Optimisation (Optuna)

### 🎯 Ce que nous avons fait
Optuna effectue une recherche automatique des meilleurs hyperparamètres.

**Actions réalisées:**
1. Créé `src/optimize_hyperparameters.py` (230 lignes)
2. Défini l'espace de recherche pour 9 hyperparamètres
3. Exécuté 30 trials (plus que le minimum de 5-10)
4. Loggé chaque trial dans MLflow
5. Sauvegardé le meilleur modèle

### 📂 Fichier: `src/optimize_hyperparameters.py`
```python
import optuna
import mlflow

def objective(trial, X_train, y_train, X_test, y_test):
    # Espace de recherche pour 9 hyperparamètres
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
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    rmse = calculate_rmse(model, X_test, y_test)
    
    # Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("test_rmse", rmse)
    
    return rmse  # Optuna minimise cette valeur

# Configuration de l'étude (30 trials)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
```

### Hyperparamètres optimisés
| Paramètre | Espace de recherche | Meilleure valeur |
|-----------|---------------------|------------------|
| learning_rate | [0.01, 0.15] | **0.05** |
| max_depth | [3, 8] | **4** |
| n_estimators | [200, 500] | **300** |
| subsample | [0.7, 1.0] | **0.85** |
| colsample_bytree | [0.6, 1.0] | **0.8** |
| min_child_weight | [1, 7] | **3** |
| gamma | [0.0, 2.0] | **0.1** |
| reg_alpha | [0.0, 3.0] | **0.5** |
| reg_lambda | [0.0, 3.0] | **1.0** |

### 🔍 Comment exécuter
```bash
python src/optimize_hyperparameters.py

# Output:
# [1/5] Loading and preprocessing data...
# [2/5] Creating Optuna study...
# [3/5] Running optimization (30 trials)...
#   Trial 1: RMSE = 22.45
#   Trial 2: RMSE = 19.87
#   ...
#   Trial 30: RMSE = 18.91
# 
# 🎯 Best RMSE: 18.64 cycles
# ✓ Improvement from baseline: 32.07 cycles (63.2% better)
```

### Résultats
```
🎯 Best RMSE: 18.64 cycles
✓ Amélioration de 63% par rapport au baseline (50.71 → 18.64)
✓ 30 trials exécutés et loggés dans MLflow
✓ Meilleur modèle sauvegardé: model_optimized.ubj
```

---

## 🚀 3.8 CI/CD (GitHub Actions)

### Pipeline CI (`ci_cd.yaml`)
```yaml
jobs:
  test-and-lint:
    - Checkout code
    - Setup Python 3.9
    - Install dependencies
    - Lint with flake8
    - Run pytest
  
  build-container:
    - Build Docker image
    - Test health endpoint
    - Push to registry
```

### Pipeline Deploy (`deploy-azure.yaml`)
```yaml
jobs:
  deploy:
    - Login to Azure
    - Push to Azure Container Registry
    - Deploy to Azure Container Apps
```

---

## 🌐 3.9 Déploiement (Serving)

### API FastAPI - Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations API |
| GET | `/health` | Health check |
| POST | `/predict` | Prédiction RUL |
| POST | `/predict/batch` | Prédiction batch |
| GET | `/model-info` | Métadonnées modèle |
| GET | `/monitoring` | Détection drift |
| GET | `/monitoring/reset` | Reset monitoring |

### Screenshot - API Swagger UI
![API Swagger UI](screenshots/api_swagger_ui.png)

### URLs de déploiement

| Service | URL | Status |
|---------|-----|--------|
| **Local Docker** | http://localhost:8000 | ✅ Running |
| **Azure Cloud** | https://aeroguard-api.salmonfield-cb3d4cec.francecentral.azurecontainerapps.io/ | ✅ Deployed |
| **Streamlit UI** | https://turbofan-predictive-m-cuczeudvjuhekghyeqtcj9.streamlit.app/ | ✅ Online |

### Screenshot - Streamlit Prediction
![Streamlit Prediction](screenshots/streamlit_prediction.png)

### Simulation v1 → v2 → Rollback
```bash
# Deploy v1
git checkout v1
docker-compose up -d --build

# Update to v2
git checkout v2
docker-compose up -d --build

# Rollback to v1
git checkout v1
docker-compose up -d --build

# Return to main
git checkout main
```

---

## 🎁 4. Bonus Implémentés

### Bonus 1: Monitoring (Drift Detection) ✅

#### 🎯 Ce que nous avons fait
Le monitoring détecte quand les nouvelles données diffèrent significativement des données d'entraînement.

**Actions réalisées:**
1. Créé l'endpoint `/monitoring` dans l'API
2. Stockage des 100 dernières prédictions en mémoire
3. Comparaison avec les statistiques baseline du training set
4. Seuil de drift: 20% de déviation

#### Code de l'endpoint (dans api/main.py)
```python
@app.get("/monitoring")
async def monitor_drift():
    # Compare recent predictions with baseline stats
    for feature in BASELINE_STATS:
        deviation = abs(recent_mean - baseline_val) / baseline_val
        if deviation > DRIFT_THRESHOLD:  # 20%
            drifted_features.append(feature)
    
    return {
        "drift_detected": len(drifted_features) > 0,
        "metrics": {...}
    }
```

#### 🔍 Comment tester le monitoring
```bash
# 1. Vérifier que l'API tourne
curl http://localhost:8000/health

# 2. Appeler l'endpoint monitoring
curl http://localhost:8000/monitoring

# Réponse attendue:
{
  "drift_detected": false,
  "status": "No data available for monitoring",
  "metrics": {},
  "recent_requests": 0
}

# 3. Faire quelques prédictions, puis re-vérifier
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
curl http://localhost:8000/monitoring
# Maintenant vous verrez les statistiques!
```

#### Script de simulation: `simulate_drift.py` (313 lignes)
```bash
python simulate_drift.py

# Phase 1: Envoie 25 requêtes normales → Pas de drift
# Phase 2: Envoie 25 requêtes corrompues (×1.5) → Drift détecté!
```

---

### Bonus 2: Retrain Automatique ✅

#### 🎯 Ce que nous avons fait
Script de réentraînement automatique déclenché par le drift ou manuellement.

**Actions réalisées:**
1. Créé `retrain.py` (266 lignes)
2. Vérifie le status de drift via l'API
3. Charge les données et réentraîne si nécessaire
4. Sauvegarde le nouveau modèle + backup de l'ancien
5. Log le retrain dans MLflow

#### Fichier: `retrain.py`
```python
def run_retrain(reason="manual"):
    # [1/4] Load training data
    X_train, y_train, X_test, y_test, feature_cols = load_training_data()
    
    # [2/4] Train model with best hyperparameters
    model = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=4, n_estimators=300, ...
    )
    model.fit(X_train, y_train)
    
    # [3/4] Save model (backup old one first)
    backup_path = f"model_backup_{timestamp}.ubj"
    model.get_booster().save_model(MODEL_PATH)
    
    # [4/4] Log to MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_artifact(MODEL_PATH)
```

#### 🔍 Comment utiliser
```bash
# Vérifier si retrain nécessaire (sans exécuter)
python retrain.py --check-only
# Output: "✓ No drift detected" ou "⚠️ Data drift detected!"

# Forcer le retrain maintenant
python retrain.py --force --reason "scheduled_weekly"
# Output:
# [1/4] Loading training data... ✓
# [2/4] Training model... ✓ RMSE: 18.64 cycles
# [3/4] Saving model... ✓
# [4/4] Logging results... ✓
# ✅ RETRAIN COMPLETE

# Retrain automatique si drift détecté
python retrain.py
# Vérifie /monitoring, puis retrain si drift_detected=true
```

---

## 📊 5. Livrables

| Livrable | Status | Fichier/URL |
|----------|--------|-------------|
| Lien GitHub | ✅ | https://github.com/AymenMB/turbofan-predictive-maintenance-mlops |
| Dockerfile | ✅ | `Dockerfile` |
| docker-compose.yml | ✅ | `docker-compose.yml` |
| Configuration DVC | ✅ | `data/raw.dvc`, `.dvc/config` |
| MLflow experiments | ✅ | `mlruns/` (4+ runs) |
| ZenML pipeline | ✅ | `pipelines/training_pipeline.py` |
| CI/CD | ✅ | `.github/workflows/*.yaml` |
| API déployée | ✅ | Azure Container Apps |
| Documentation | ✅ | `README.md`, `MLOPS_STEP_BY_STEP.md` |

---

## 🚀 Quick Start

### 1. Cloner le repository
```bash
git clone https://github.com/AymenMB/turbofan-predictive-maintenance-mlops.git
cd turbofan-predictive-maintenance-mlops
```

### 2. Installer les dépendances
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Récupérer les données
```bash
dvc pull
```

### 4. Lancer l'API avec Docker
```bash
docker-compose up -d
```

### 5. Tester l'API
```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### 6. Ouvrir Swagger UI
```
http://localhost:8000/docs
```

---

## 📈 Résultats

### Performance du modèle

| Métrique | Baseline | Optimisé | Amélioration |
|----------|----------|----------|--------------|
| RMSE | 50.71 | **18.64** | -63% |
| R² | 0.56 | **0.79** | +41% |

### Architecture déployée

```
┌─────────────────────────────────────────────────────────────────┐
│                        AZURE CLOUD                              │
│  ┌──────────────────┐      ┌──────────────────────────────┐    │
│  │ Container        │      │ Azure Container Apps         │    │
│  │ Registry (ACR)   │─────▶│  aeroguard-api              │    │
│  │ aeroguardacr     │      │  FastAPI + XGBoost          │    │
│  └──────────────────┘      └──────────────────────────────┘    │
│                                         │                       │
└─────────────────────────────────────────┼───────────────────────┘
                                          │ HTTPS
                  ┌───────────────────────┼───────────────────────┐
                  │    STREAMLIT CLOUD    │                       │
                  │  ┌────────────────────▼────────────────────┐  │
                  │  │   AeroGuard AI Dashboard               │  │
                  │  │   - Predict RUL                        │  │
                  │  │   - Batch Analysis                     │  │
                  │  └─────────────────────────────────────────┘  │
                  └───────────────────────────────────────────────┘
```

---

## 📚 Documentation Additionnelle

- [MLOPS_STEP_BY_STEP.md](MLOPS_STEP_BY_STEP.md) - Guide détaillé pas à pas
- [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md) - Guide de déploiement Azure
- [DOCUMENTATION.md](DOCUMENTATION.md) - Documentation technique

---

## 🏆 Conclusion

Ce projet implémente **tous les 9 requirements** du cahier des charges Mini-projet MLOps ainsi que les **2 bonus optionnels** (Monitoring et Retrain automatique).

### ✅ Checklist finale

- [x] 3.1 Dataset public (NASA C-MAPSS) + Modèle baseline (XGBoost)
- [x] 3.2 Git avec branches (main/dev) et tags (v1/v2/v3)
- [x] 3.3 Docker + Docker Compose
- [x] 3.4 DVC pour versioning des données
- [x] 3.5 MLflow pour experiment tracking
- [x] 3.6 ZenML pour pipeline orchestration
- [x] 3.7 Optuna pour optimisation (30 trials)
- [x] 3.8 CI/CD avec GitHub Actions
- [x] 3.9 API déployée sur Azure + simulation v1→v2→rollback
- [x] **Bonus 1:** Monitoring (drift detection)
- [x] **Bonus 2:** Retrain automatique

---

<div align="center">

**Réalisé par Aymen MABROUK**

*Sous la supervision de Dr. Salah GONTARA*

École Polytechnique Sousse | 2025-2026

</div>
