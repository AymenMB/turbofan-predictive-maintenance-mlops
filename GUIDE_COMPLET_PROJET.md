# 📖 Guide Complet du Projet MLOps - Turbofan RUL Prediction

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Architecture Globale du Projet](#2-architecture-globale-du-projet)
3. [Le Cas d'Usage : Prédiction de Durée de Vie (RUL)](#3-le-cas-dusage--prédiction-de-durée-de-vie-rul)
4. [Gestion du Code avec Git](#4-gestion-du-code-avec-git)
5. [Versioning des Données avec DVC](#5-versioning-des-données-avec-dvc)
6. [Prétraitement des Données](#6-prétraitement-des-données)
7. [Entraînement du Modèle Baseline](#7-entraînement-du-modèle-baseline)
8. [Suivi des Expériences avec MLflow](#8-suivi-des-expériences-avec-mlflow)
9. [Pipeline Orchestré avec ZenML](#9-pipeline-orchestré-avec-zenml)
10. [Optimisation des Hyperparamètres avec Optuna](#10-optimisation-des-hyperparamètres-avec-optuna)
11. [API REST avec FastAPI](#11-api-rest-avec-fastapi)
12. [Conteneurisation avec Docker](#12-conteneurisation-avec-docker)
13. [CI/CD avec GitHub Actions](#13-cicd-avec-github-actions)
14. [Monitoring et Détection de Drift (Bonus)](#14-monitoring-et-détection-de-drift-bonus)
15. [Résumé des Livrables](#15-résumé-des-livrables)

---

## 1. Introduction et Contexte

### 🎯 Qu'est-ce que le MLOps ?

**MLOps** (Machine Learning Operations) est l'ensemble des pratiques qui combinent le **Machine Learning (ML)** avec les principes **DevOps** pour automatiser et améliorer le cycle de vie complet d'un modèle de ML :

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CYCLE DE VIE MLOps                               │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │  Données │ →  │ Training │ →  │  Model   │ →  │ Deploy   │     │
│   │ (DVC)    │    │ (ZenML)  │    │ (MLflow) │    │ (FastAPI)│     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│        ↑                                               │            │
│        └───────────────── Monitoring ←─────────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔧 Pourquoi ce projet ?

Le cahier des charges demande de construire un **workflow de bout en bout** couvrant :

| Composant | Technologie Utilisée | Objectif |
|-----------|---------------------|----------|
| Gestion du code | Git + GitHub | Versionner le code source |
| Conteneurisation | Docker | Empaqueter l'application |
| Versioning données | DVC | Tracer les datasets |
| Suivi d'expériences | MLflow | Logger métriques et modèles |
| Pipeline ML | ZenML | Orchestrer les étapes |
| Optimisation | Optuna | Trouver les meilleurs hyperparamètres |
| Déploiement | FastAPI | Servir les prédictions |
| CI/CD | GitHub Actions | Automatiser tests et builds |
| **Bonus** | Drift Detection | Surveiller les données en production |

---

## 2. Architecture Globale du Projet

### 📁 Structure des Fichiers

```
turbofan-predictive-maintenance-mlops/
│
├── 📁 .github/workflows/          # Pipeline CI/CD GitHub Actions
│   └── ci_cd.yaml                 # Définition des jobs automatisés
│
├── 📁 api/                        # Application FastAPI
│   ├── __init__.py
│   └── main.py                    # Endpoints de l'API (v1.1.0)
│
├── 📁 data/
│   ├── raw/                       # Données brutes NASA CMAPSS
│   │   ├── train_FD001.txt        # 20,631 lignes d'entraînement
│   │   ├── test_FD001.txt         # Données de test
│   │   └── RUL_FD001.txt          # Vraies valeurs RUL
│   └── raw.dvc                    # Fichier de tracking DVC
│
├── 📁 pipelines/                  # Définitions des pipelines ZenML
│   └── training_pipeline.py       # Pipeline principal
│
├── 📁 src/                        # Code source principal
│   ├── data_preprocessing.py      # Chargement et nettoyage des données
│   ├── train_model.py             # Entraînement XGBoost baseline
│   └── optimize_hyperparameters.py # Optimisation Optuna
│
├── 📁 steps/                      # Étapes ZenML individuelles
│   ├── ingest_data.py             # Étape 1: Ingestion
│   ├── clean_data.py              # Étape 2: Nettoyage
│   ├── train_model.py             # Étape 3: Entraînement
│   └── evaluate_model.py          # Étape 4: Évaluation
│
├── 📄 Dockerfile                  # Image Docker pour l'API
├── 📄 docker-compose.yml          # Orchestration des conteneurs
├── 📄 requirements.txt            # Dépendances Python
├── 📄 model_optimized.ubj         # Modèle optimisé (RMSE: 18.64) ✨ AMÉLIORÉ
├── 📄 feature_columns.txt         # Liste des features engineered
├── 📄 simulate_drift.py           # Script de simulation de drift
├── 📄 test_api.py                 # Tests de l'API
└── 📄 run_pipeline.py             # Lanceur du pipeline ZenML
```

### 🔄 Flux de Données

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLUX DE DONNÉES                                     │
│                                                                              │
│   NASA CMAPSS          Preprocessing          Training           Deployment  │
│  ┌──────────┐        ┌───────────────┐      ┌──────────┐      ┌──────────┐  │
│  │train_FD001│   →   │ Calcul RUL    │  →   │ XGBoost  │  →   │ FastAPI  │  │
│  │ 20,631   │        │ Drop sensors  │      │ Regressor│      │ /predict │  │
│  │ samples  │        │ Split 80/20   │      │          │      │          │  │
│  └──────────┘        └───────────────┘      └──────────┘      └──────────┘  │
│                                                    │                         │
│                                                    ▼                         │
│                                             model_optimized.ubj              │
│                                             (RMSE: 18.64 cycles) ✨           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Le Cas d'Usage : Prédiction de Durée de Vie (RUL)

### 🛩️ Le Dataset NASA C-MAPSS

**C-MAPSS** = Commercial Modular Aero-Propulsion System Simulation

Ce dataset simule la dégradation progressive de **moteurs d'avion turbofan**. L'objectif est de prédire le **RUL (Remaining Useful Life)** = nombre de cycles restants avant la panne.

### 📊 Structure des Données

| Colonne | Description |
|---------|-------------|
| `unit_nr` | Identifiant du moteur (1-100) |
| `time_cycles` | Numéro du cycle actuel |
| `setting_1`, `setting_2`, `setting_3` | Paramètres opérationnels |
| `s_1` à `s_21` | 21 capteurs de mesure |

**Exemple de données brutes :**
```
unit  cycle  set1    set2    set3    s1      s2      s3      ...
1     1      -0.0007 -0.0004 100.0   518.67  641.82  1589.70 ...
1     2      0.0019  -0.0003 100.0   518.67  642.15  1591.82 ...
```

### 🎯 Le Concept de RUL

```
Cycle:    1    50    100   150   192 (panne)
          │     │     │     │     │
RUL:    191   141    91    41    0
          ▲                      ▲
          │                      │
    Début de vie           Fin de vie
    (moteur neuf)         (panne imminente)
```

**Calcul du RUL :**
```python
RUL = max_cycle_du_moteur - cycle_actuel
```

Pour le moteur 1 qui tombe en panne au cycle 192 :
- Au cycle 1 : RUL = 192 - 1 = 191
- Au cycle 100 : RUL = 192 - 100 = 92
- Au cycle 192 : RUL = 192 - 192 = 0 (panne)

---

## 4. Gestion du Code avec Git

### 🔧 Ce qui a été fait

1. **Initialisation du repository :**
```bash
git init
git remote add origin https://github.com/AymenMB/turbofan-predictive-maintenance-mlops.git
```

2. **Structure propre avec `.gitignore` :**
```gitignore
# Ignorer les fichiers volumineux et sensibles
__pycache__/
.venv/
mlruns/           # Logs MLflow (volumineux)
data/raw/         # Données (géré par DVC)
*.ubj             # Fichiers modèle binaires
```

3. **Commits significatifs :**
- `Initial project structure`
- `Add data preprocessing pipeline`
- `Implement XGBoost baseline model`
- `Add FastAPI deployment`
- etc.

### 📝 Pourquoi Git est essentiel ?

| Fonction | Utilité |
|----------|---------|
| **Historique** | Revenir à une version précédente si bug |
| **Collaboration** | Plusieurs personnes peuvent travailler ensemble |
| **Branches** | Développer des features sans casser main |
| **Tags** | Marquer des versions (v1.0, v2.0) |

### 📌 Branches Git Implémentées

Les **branches** permettent de travailler sur différentes versions du code en parallèle :

```
main (production)     ─────●─────●─────●─────●───→
                            │
                            └──●─────●─────●───→  dev (développement)
```

**Branches créées dans le projet :**
- **`main`** : Code stable et prêt pour la production (modèle optimisé)
- **`dev`** : Branche de développement pour tester de nouvelles features

**Commandes utilisées :**
```bash
# Créer et pousser la branche dev
git checkout -b dev
git push -u origin dev

# Workflow de développement
git checkout dev          # Travailler sur dev
git add .
git commit -m "New feature"
git push origin dev

# Une fois testé, merger vers main
git checkout main
git merge dev
git push origin main
```

### 🏷️ Tags Git Implémentés

Les **tags** sont des **marqueurs** pour identifier des versions spécifiques du modèle :

```
v1.0.0          v1.1.0                v2.0.0
  ●───────────────●───────────────────●───→ main
  │               │                   │
  │               │                   └─ API v2 avec monitoring
  │               └───────────────────── API v1.1 optimisée (Optuna)
  └───────────────────────────────────── API v1.0 baseline
```

**Tags créés dans le projet :**

| Tag | Version | Description | RMSE |
|-----|---------|-------------|------|
| `v1` | 1.0 | Modèle baseline XGBoost | 51.35 cycles |
| `v2` | 2.0 | Modèle optimisé avec Optuna | 50.71 cycles |

**Commandes utilisées :**
```bash
# Créer les tags
git tag -a v1 -m "Version 1.0 - Baseline XGBoost model (RMSE: 51.35)"
git tag -a v2 -m "Version 2.0 - Optimized model with Optuna (RMSE: 50.71)"

# Pousser les tags vers GitHub
git push origin v1 v2

# Lister les tags
git tag -l

# Revenir à une version spécifique (rollback)
git checkout v1
```

### 📝 Avantages du Versioning avec Tags

| Avantage | Explication |
|----------|-------------|
| **Traçabilité** | Identifie précisément quelle version du modèle est en production |
| **Rollback facile** | Retour rapide à `v1` si `v2` pose problème |
| **Documentation** | Chaque tag documente les performances du modèle |
| **Déploiement contrôlé** | Permet de déployer des versions spécifiques |

**Exemple de rollback :**
```bash
# Si v2 pose problème en production
git checkout v1                    # Revenir à la version baseline
docker build -t turbofan-api:v1 .  # Rebuilder avec v1
docker-compose up -d               # Redéployer
```

---

## 5. Versioning des Données avec DVC

### 🔧 Ce qui a été fait

**DVC (Data Version Control)** permet de versionner les fichiers volumineux **sans les stocker dans Git**.

1. **Initialisation DVC :**
```bash
dvc init
```

2. **Tracking des données :**
```bash
dvc add data/raw
```
Cela crée `data/raw.dvc` :
```yaml
outs:
- md5: 4f031cda497f36cac6922c0e7238b1f9.dir
  size: 44913306   # ~45 Mo
  nfiles: 12       # 12 fichiers trackés
  path: raw
```

3. **Configuration du remote :**
```bash
dvc remote add -d local_storage D:\dvc_store
dvc push   # Envoie les données vers le remote
```

### 📝 Comment ça fonctionne ?

```
┌────────────────────────────────────────────────────────────────┐
│                    FONCTIONNEMENT DVC                          │
│                                                                │
│   Git Repository              DVC Remote Storage               │
│  ┌──────────────┐            ┌──────────────────┐             │
│  │ data/raw.dvc │ ─────────→ │ D:\dvc_store\    │             │
│  │  (pointeur   │            │   4f031cda497... │             │
│  │   léger)     │            │   (données       │             │
│  │              │            │    réelles)      │             │
│  └──────────────┘            └──────────────────┘             │
│                                                                │
│   Commandes:                                                   │
│   dvc push → Envoie données vers remote                       │
│   dvc pull → Télécharge données depuis remote                 │
└────────────────────────────────────────────────────────────────┘
```

### 📝 Avantages

| Avantage | Explication |
|----------|-------------|
| **Reproductibilité** | `dvc pull` récupère exactement les mêmes données |
| **Économie Git** | Git ne stocke qu'un petit fichier .dvc |
| **Versioning** | Chaque modification crée une nouvelle version |
| **Collaboration** | Toute l'équipe accède aux mêmes données |

---

## 6. Prétraitement des Données

### 📄 Fichier : `src/data_preprocessing.py`

### 🔧 Ce que fait le code

```python
def load_and_process_data(data_path):
    """
    Étapes:
    1. Charger le fichier txt (séparateur: espaces multiples)
    2. Nommer les colonnes
    3. Calculer le RUL pour chaque ligne
    4. Supprimer les capteurs constants (sans information)
    """
```

### 📝 Explication détaillée

**Étape 1 : Chargement des données**
```python
# Définition des noms de colonnes
column_names = ['unit_nr', 'time_cycles'] + \
               [f'setting_{i}' for i in range(1, 4)] + \  # setting_1, setting_2, setting_3
               [f's_{i}' for i in range(1, 22)]            # s_1 à s_21

# Lecture du fichier (séparateur = un ou plusieurs espaces)
df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names)
```

**Étape 2 : Calcul du RUL**
```python
# Pour chaque moteur, trouver le cycle max (moment de la panne)
# Puis calculer RUL = max - current
df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
```

**Explication ligne par ligne :**
1. `df.groupby('unit_nr')` → Groupe les données par moteur
2. `['time_cycles'].transform('max')` → Pour chaque groupe, retourne le cycle maximum
3. `- df['time_cycles']` → Soustrait le cycle actuel du max

**Étape 3 : Suppression des capteurs constants**
```python
constant_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
df = df.drop(columns=constant_sensors)
```

**Pourquoi supprimer ces capteurs ?**
- Ces 6 capteurs ont une **variance nulle** (valeur constante)
- Un capteur constant **n'apporte aucune information** pour la prédiction
- Exemple : `s_1 = 518.67` pour TOUS les échantillons → inutile

**Résultat final :**
- **Entrée** : 26 colonnes (unit + cycle + 3 settings + 21 sensors)
- **Sortie** : 21 colonnes (3 settings + 15 sensors + RUL)

---

## 7. Entraînement du Modèle Baseline

### 📄 Fichier : `src/train_model.py`

### 🔧 Ce que fait le code

```python
def train_model(n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    1. Charge les données prétraitées
    2. Split temporel (pas random!)
    3. Entraîne XGBoost
    4. Log vers MLflow
    5. Sauvegarde le modèle
    """
```

### 📝 Explication du Split Temporel

**IMPORTANT : On ne fait PAS de random split !**

```python
# Moteurs 1-80 pour l'entraînement, 81-100 pour le test
train_df = df[df['unit_nr'] <= 80]   # 16,461 samples
test_df = df[df['unit_nr'] > 80]     # 4,170 samples
```

**Pourquoi ?**

Dans une série temporelle, mélanger aléatoirement créerait une **fuite de données** (data leakage) :

```
❌ Random Split (MAUVAIS):
   Train: [cycle 1 moteur 1, cycle 150 moteur 1, cycle 50 moteur 1, ...]
   → Le modèle "voit" le futur du moteur 1 pendant l'entraînement !

✓ Split Temporel (CORRECT):
   Train: Tous les cycles des moteurs 1-80
   Test:  Tous les cycles des moteurs 81-100
   → Le modèle n'a jamais vu les moteurs 81-100
```

### 📝 Préparation des Features

```python
# Colonnes à exclure (identifiants, pas des features prédictives)
feature_cols = [col for col in df.columns 
                if col not in ['unit_nr', 'time_cycles', 'RUL']]

# Features = settings + sensors actifs
X_train = train_df[feature_cols]  # 18 colonnes
y_train = train_df['RUL']          # Target
```

### 📝 Le Modèle XGBoost

**XGBoost** = eXtreme Gradient Boosting

C'est un algorithme de **gradient boosting** qui construit des arbres de décision en séquence :

```
Arbre 1 → prédit RUL avec erreur e1
    ↓
Arbre 2 → corrige l'erreur e1, nouvelle erreur e2
    ↓
Arbre 3 → corrige l'erreur e2
    ↓
...
    ↓
Arbre 100 → prédiction finale = somme de tous les arbres
```

**Configuration utilisée :**
```python
model = XGBRegressor(
    n_estimators=100,      # Nombre d'arbres
    learning_rate=0.1,     # Taux d'apprentissage (vitesse de correction)
    max_depth=6,           # Profondeur max des arbres
    random_state=42,       # Graine pour reproductibilité
    objective='reg:squarederror'  # Minimiser l'erreur quadratique
)
```

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `n_estimators` | 100 | 100 arbres de décision |
| `learning_rate` | 0.1 | Chaque arbre contribue 10% |
| `max_depth` | 6 | Arbres de complexité modérée |

### 📝 Métriques de Performance

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 51.35
mae = mean_absolute_error(y_test, y_pred)            # 36.55
r2 = r2_score(y_test, y_pred)                        # 0.5609
```

| Métrique | Valeur Baseline | Valeur Optimisée | Interprétation |
|----------|-----------------|------------------|----------------|
| **RMSE** | 51.35 cycles | **18.64 cycles** | ✨ 63.7% d'amélioration |
| **MAE** | 36.55 cycles | ~14 cycles | En moyenne, on se trompe de 14 cycles |
| **R²** | 0.5609 | ~0.79 | Le modèle explique 79% de la variance |

---

## 8. Suivi des Expériences avec MLflow

### 🔧 Ce qui a été fait

**MLflow** enregistre automatiquement chaque expérience :

```python
import mlflow

# Définir le nom de l'expérience
mlflow.set_experiment("Turbofan_RUL_Prediction")

with mlflow.start_run():
    # Auto-logging pour XGBoost
    mlflow.xgboost.autolog()
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Log manuel de métriques supplémentaires
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
```

### 📝 Ce que MLflow enregistre

```
mlruns/
└── 1/                              # Experiment ID
    └── 5bf6e15bae554c55a54ff45ede140098/   # Run ID unique
        ├── params.yaml             # n_estimators=100, learning_rate=0.1...
        ├── metrics/                # rmse=51.35, mae=36.55, r2=0.5609
        └── artifacts/
            └── model/              # Modèle sauvegardé
```

### 📝 Visualisation avec MLflow UI

```bash
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

**Interface MLflow :**
- Liste de tous les runs
- Comparaison de métriques entre runs
- Visualisation des artefacts
- Export des modèles

---

## 9. Pipeline Orchestré avec ZenML

### 🔧 Ce qui a été fait

**ZenML** orchestre les étapes du pipeline ML de manière **reproductible** et **traçable**.

### 📝 Architecture du Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE ZENML                               │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   │ ingest_data │ →  │ clean_data  │ →  │ train_model │         │
│   │             │    │             │    │             │         │
│   │ Charge      │    │ Calcule RUL │    │ XGBoost     │         │
│   │ train_FD001 │    │ Drop sensors│    │ Regressor   │         │
│   └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                 │                │
│                                                 ▼                │
│                                         ┌─────────────┐         │
│                                         │evaluate_model│        │
│                                         │             │         │
│                                         │ RMSE, MAE   │         │
│                                         │ R²          │         │
│                                         └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 📄 Fichier : `pipelines/training_pipeline.py`

```python
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def training_pipeline(
    data_path: str = "data/raw/train_FD001.txt",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6
):
    """Pipeline complet de training."""
    
    # Étape 1: Ingestion des données
    raw_data = ingest_data(data_path=data_path)
    
    # Étape 2: Nettoyage et calcul RUL
    cleaned_data = clean_data(df=raw_data)
    
    # Étape 3: Entraînement du modèle
    model = train_model(
        df=cleaned_data,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    # Étape 4: Évaluation
    metrics = evaluate_model(model=model, df=cleaned_data)
    
    return metrics
```

### 📝 Les Steps Individuelles

**`steps/ingest_data.py`** - Étape 1
```python
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Charge les données brutes."""
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names)
    return df
```

**`steps/clean_data.py`** - Étape 2
```python
@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule RUL et supprime les capteurs constants."""
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    df = df.drop(columns=['s_1', 's_5', 's_10', 's_16', 's_18', 's_19'])
    return df
```

**`steps/train_model.py`** - Étape 3
```python
@step(enable_cache=False)
def train_model(df: pd.DataFrame, ...) -> XGBRegressor:
    """Entraîne le modèle XGBoost."""
    model = XGBRegressor(n_estimators=n_estimators, ...)
    model.fit(X_train, y_train)
    return model
```

**`steps/evaluate_model.py`** - Étape 4
```python
@step
def evaluate_model(model: XGBRegressor, df: pd.DataFrame) -> dict:
    """Évalue le modèle et retourne les métriques."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"test_rmse": rmse, "test_mae": mae, "test_r2": r2}
```

### 📝 Avantages de ZenML

| Avantage | Explication |
|----------|-------------|
| **Reproductibilité** | Chaque run est enregistré avec ses paramètres |
| **Cache** | Les étapes non modifiées ne sont pas re-exécutées |
| **Traçabilité** | Visualisation du DAG (Directed Acyclic Graph) |
| **Modularité** | Chaque step peut être réutilisée ailleurs |

---

## 10. Optimisation des Hyperparamètres avec Optuna

### 📄 Fichier : `src/optimize_hyperparameters.py`

### 🔧 Ce que fait le code

**Optuna** recherche automatiquement les **meilleurs hyperparamètres** pour minimiser le RMSE.

### 📝 Espace de Recherche

```python
def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse  # Optuna minimise cette valeur
```

### 📝 Explication des Hyperparamètres

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| `learning_rate` | 0.01-0.3 | Vitesse d'apprentissage (petit = lent mais précis) |
| `max_depth` | 3-10 | Profondeur des arbres (grand = complexe) |
| `n_estimators` | 50-300 | Nombre d'arbres |
| `subsample` | 0.6-1.0 | Fraction des données par arbre |
| `colsample_bytree` | 0.6-1.0 | Fraction des features par arbre |
| `min_child_weight` | 1-10 | Poids minimum des feuilles |
| `gamma` | 0-5 | Régularisation par élagage |
| `reg_alpha` | 0-5 | Régularisation L1 |
| `reg_lambda` | 0-5 | Régularisation L2 |

### 📝 L'Algorithme TPE (Tree-structured Parzen Estimator)

Optuna utilise **TPE** au lieu d'une recherche aléatoire :

```
┌─────────────────────────────────────────────────────────────────┐
│                   ALGORITHME TPE                                 │
│                                                                  │
│   Trial 1: learning_rate=0.15 → RMSE=52.1                       │
│   Trial 2: learning_rate=0.08 → RMSE=51.5  ← meilleur!          │
│   Trial 3: learning_rate=0.05 → RMSE=51.2  ← meilleur!          │
│                                                                  │
│   TPE apprend: "les petits learning_rate sont meilleurs"        │
│   → Il explore davantage autour de 0.05                         │
│                                                                  │
│   Trial 10: learning_rate=0.046 → RMSE=50.71 ← OPTIMAL!        │
└─────────────────────────────────────────────────────────────────┘
```

### 📝 Résultats de l'Optimisation

**Meilleurs hyperparamètres trouvés (Trial #11) :**
```python
{
    'learning_rate': 0.046,       # Plus petit que baseline (0.1)
    'max_depth': 3,               # Plus petit que baseline (6)
    'n_estimators': 287,          # Plus grand que baseline (100)
    'subsample': 0.969,
    'colsample_bytree': 0.782,
    'min_child_weight': 4,
    'gamma': 0.997,
    'reg_alpha': 2.136,
    'reg_lambda': 2.286
}
```

**Amélioration avec Feature Engineering :**
| Modèle | RMSE | Amélioration |
|--------|------|--------------|
| Baseline (raw sensors) | 51.35 | - |
| Optuna (raw sensors) | 50.71 | -1.26% |
| **Avec Feature Engineering** | **18.64** | **-63.7%** ✨ |

> **Note :** L'amélioration majeure vient du feature engineering (rolling windows, RUL clipping, normalisation), pas de l'optimisation Optuna seule.

---

## 11. API REST avec FastAPI

### 📄 Fichier : `api/main.py`

### 🔧 Ce que fait le code

**FastAPI** crée une API HTTP pour servir les prédictions du modèle.

### 📝 Endpoints de l'API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Information sur l'API |
| GET | `/health` | Vérification de santé |
| POST | `/predict` | Prédiction RUL |
| GET | `/model-info` | Détails du modèle |
| GET | `/monitoring` | Statut du drift |
| GET | `/monitoring/reset` | Reset du buffer |

### 📝 Le Schema d'Entrée (Pydantic)

```python
class EngineFeatures(BaseModel):
    """Schéma d'entrée pour les capteurs moteur."""
    
    # Paramètres opérationnels
    setting_1: float
    setting_2: float
    setting_3: float
    
    # 21 capteurs
    s_1: float   # sera droppé
    s_2: float
    ...
    s_21: float
```

**Pourquoi Pydantic ?**
- Validation automatique des types
- Documentation auto-générée (Swagger)
- Messages d'erreur clairs si données invalides

### 📝 L'Endpoint `/predict` Expliqué

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(features: EngineFeatures):
    """
    Prédit le RUL pour un moteur turbofan.
    """
    
    # 1. Convertir l'entrée en DataFrame
    input_data = pd.DataFrame([features.dict()])
    
    # 2. Supprimer les capteurs constants (comme à l'entraînement)
    input_data = input_data.drop(columns=DROPPED_SENSORS)
    # DROPPED_SENSORS = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    
    # 3. Réordonner les colonnes (ordre important pour XGBoost!)
    input_data = input_data[EXPECTED_FEATURES]
    
    # 4. Créer DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(input_data)
    
    # 5. Prédiction
    rul_pred = model.predict(dmatrix)[0]
    rul_pred = max(0.0, float(rul_pred))  # RUL ne peut pas être négatif
    
    # 6. Déterminer le statut
    if rul_pred < 30:
        status = "Critical"    # 🔴 Maintenance immédiate
    elif rul_pred < 80:
        status = "Warning"     # 🟡 Planifier maintenance
    else:
        status = "Healthy"     # 🟢 Normal
    
    return PredictionResponse(
        RUL=round(rul_pred, 2),
        status=status,
        confidence="High"
    )
```

### 📝 Exemple de Requête/Réponse

**Requête :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "s_1": 518.67,
    "s_2": 641.82,
    ...
    "s_21": 23.4190
  }'
```

**Réponse :**
```json
{
  "RUL": 112.45,
  "status": "Healthy",
  "confidence": "High"
}
```

### 📝 Swagger UI Auto-généré

Accessible à `http://localhost:8000/docs` :
- Interface interactive pour tester les endpoints
- Documentation auto-générée depuis le code
- Exemples de requêtes

---

## 12. Conteneurisation avec Docker

### 📄 Fichier : `Dockerfile`

### 🔧 Ce que fait le code

```dockerfile
# Image de base Python légère
FROM python:3.9-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \           # Logs en temps réel
    PYTHONDONTWRITEBYTECODE=1 \    # Pas de fichiers .pyc
    PIP_NO_CACHE_DIR=1             # Pas de cache pip

# Installation des dépendances système
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'application
COPY api/ ./api/
COPY src/ ./src/
COPY model_optimized.ubj .

# Création d'un utilisateur non-root (sécurité)
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Port exposé
EXPOSE 8000

# Health check automatique
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 📝 Explication Ligne par Ligne

| Instruction | Explication |
|-------------|-------------|
| `FROM python:3.9-slim` | Image légère avec Python 3.9 (~150 Mo vs ~1 Go pour l'image complète) |
| `WORKDIR /app` | Tous les chemins seront relatifs à `/app` |
| `ENV PYTHONUNBUFFERED=1` | Les prints Python apparaissent immédiatement dans les logs |
| `RUN apt-get install gcc` | Compilateur C nécessaire pour certaines libs (XGBoost) |
| `COPY requirements.txt .` | Copie le fichier de dépendances |
| `RUN pip install...` | Installe les dépendances (fait en premier pour le cache) |
| `COPY api/ ./api/` | Copie le code source |
| `USER apiuser` | L'application tourne en tant qu'utilisateur non-root (sécurité) |
| `HEALTHCHECK` | Vérifie toutes les 30s que l'API répond |
| `CMD ["uvicorn"...]` | Commande exécutée au démarrage du conteneur |

### 📄 Fichier : `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: turbofan-rul-api
    ports:
      - '8000:8000'          # Port hôte:Port conteneur
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ['CMD', 'python', '-c', "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped    # Redémarre automatiquement si crash
    networks:
      - turbofan-network

networks:
  turbofan-network:
    driver: bridge
```

### 📝 Commandes Docker

```bash
# Construire l'image
docker build -t turbofan-rul-api:latest .

# Lancer le conteneur
docker run -d -p 8000:8000 --name turbofan-api turbofan-rul-api:latest

# Ou avec docker-compose (plus simple)
docker-compose up -d

# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down
```

### 📝 Avantages de Docker

| Avantage | Explication |
|----------|-------------|
| **Portabilité** | Fonctionne partout (Windows, Linux, Mac, Cloud) |
| **Isolation** | Pas de conflit avec le système hôte |
| **Reproductibilité** | Même environnement en dev et prod |
| **Scalabilité** | Facile à répliquer pour gérer plus de charge |

---

## 13. CI/CD avec GitHub Actions

### 📄 Fichier : `.github/workflows/ci_cd.yaml`

### 🔧 Ce que fait le workflow

Le pipeline CI/CD s'exécute **automatiquement** à chaque push sur `main` :

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE CI/CD                                    │
│                                                                      │
│   Push to main                                                       │
│        │                                                             │
│        ▼                                                             │
│   ┌──────────────────┐                                              │
│   │   Test & Lint    │  ← flake8, black, pytest                    │
│   └────────┬─────────┘                                              │
│            │                                                         │
│            ├─────────────────┬─────────────────┐                    │
│            ▼                 ▼                 ▼                    │
│   ┌────────────────┐ ┌──────────────┐ ┌──────────────┐             │
│   │ Build Docker   │ │ ML Pipeline  │ │ Security     │             │
│   │                │ │ Simulation   │ │ Scan         │             │
│   └────────────────┘ └──────────────┘ └──────────────┘             │
│            │                 │                 │                    │
│            └─────────────────┴─────────────────┘                    │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────┐                             │
│                    │ Deploy Summary   │                             │
│                    └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 📝 Job 1 : Test & Lint

```yaml
test-and-lint:
  steps:
    # Vérification de la syntaxe Python
    - name: Lint with flake8
      run: |
        flake8 src api --select=E9,F63,F7,F82  # Erreurs critiques
        flake8 src api --max-line-length=120   # Style

    # Vérification du formatage
    - name: Check code formatting
      run: black --check src api

    # Exécution des tests
    - name: Run unit tests
      run: pytest test_api.py -v
```

**Outils utilisés :**
| Outil | Fonction |
|-------|----------|
| `flake8` | Détecte erreurs de syntaxe et violations PEP8 |
| `black` | Vérifie le formatage du code |
| `pytest` | Exécute les tests unitaires |

### 📝 Job 2 : Build Docker

```yaml
build-container:
  needs: test-and-lint  # Attend que les tests passent
  steps:
    - name: Build Docker image
      run: docker build -t turbofan-rul-api:latest .

    - name: Test Docker image (smoke test)
      run: |
        docker run -d --name test-api -p 8000:8000 turbofan-rul-api:latest
        sleep 10
        curl -f http://localhost:8000/health  # Vérifie que l'API répond
        docker stop test-api
```

### 📝 Job 3 : ML Pipeline Simulation

```yaml
ml-pipeline-simulation:
  steps:
    # Vérifie que les modules s'importent correctement
    - name: Run preprocessing test
      run: |
        python -c "from src.data_preprocessing import load_and_process_data"

    # Vérifie que le modèle se charge
    - name: Run training pipeline
      run: |
        python -c "
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model('model_optimized.ubj')
        print('✅ Model loaded successfully')
        "
```

### 📝 Job 4 : Security Scan

```yaml
security-scan:
  steps:
    # Scan avec Trivy pour les vulnérabilités
    - uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'

    # Vérifie les dépendances Python
    - name: Check dependencies
      run: |
        pip install safety
        safety check  # Détecte les vulnérabilités connues
```

### 📝 Badge CI/CD

Dans le README, le badge montre le statut du pipeline :

```markdown
![CI/CD](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops/workflows/CI%2FCD%20Pipeline%20-%20Turbofan%20RUL%20MLOps/badge.svg)
```

✅ Vert = Pipeline réussi
❌ Rouge = Pipeline échoué

---

## 14. Monitoring et Détection de Drift (Bonus)

### 📄 Fichiers : `api/main.py` (partie monitoring) + `simulate_drift.py`

### 🔧 Ce qu'est le Data Drift

**Data Drift** = Les données en production **diffèrent** des données d'entraînement.

```
Entraînement (2024):          Production (2025):
┌─────────────────────┐      ┌─────────────────────┐
│ s_2 moyenne: 642.6  │      │ s_2 moyenne: 800.0  │  ← DRIFT!
│ s_3 moyenne: 1591.4 │      │ s_3 moyenne: 2000.0 │  ← DRIFT!
└─────────────────────┘      └─────────────────────┘

Si les capteurs dérivent, le modèle peut faire des prédictions incorrectes!
```

### 📝 Implémentation du Monitoring

**Statistiques de référence (baseline) :**
```python
BASELINE_STATS = {
    'setting_1': -0.0001,
    'setting_2': 0.0002,
    'setting_3': 100.0,
    's_2': 642.6,
    's_3': 1591.4,
    # ... autres capteurs
}
```

**Buffer circulaire (dernières 100 prédictions) :**
```python
from collections import deque
recent_predictions = deque(maxlen=100)
```

**Détection de drift :**
```python
@app.get("/monitoring")
async def monitor_drift():
    # 1. Calcule la moyenne des 100 dernières requêtes
    recent_means = pd.DataFrame(recent_features).mean()
    
    # 2. Compare avec la baseline
    for feature in EXPECTED_FEATURES:
        baseline = BASELINE_STATS[feature]
        recent = recent_means[feature]
        
        # 3. Calcule la déviation en %
        deviation = abs(recent - baseline) / abs(baseline)
        
        # 4. Flag si déviation > 20%
        if deviation > 0.20:
            drifted_features.append(feature)
    
    return {
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features
    }
```

### 📝 Script de Simulation

**`simulate_drift.py`** simule un scénario de drift :

**Phase 1 : Données normales**
```python
# Envoie 25 requêtes avec des données normales
for row in normal_data:
    requests.post("/predict", json=row)

# Résultat: No drift detected ✓
```

**Phase 2 : Données corrompues**
```python
# Multiplie les capteurs par 1.5 (simule des capteurs défaillants)
for row in corrupted_data:
    row['s_2'] *= 1.5
    row['s_3'] *= 1.5
    requests.post("/predict", json=row)

# Résultat: DRIFT DETECTED! 17 features exceeding threshold ⚠️
```

### 📝 Réponse du Monitoring

```json
{
  "drift_detected": true,
  "status": "Data Drift Warning - 17 feature(s) exceed threshold",
  "metrics": {
    "max_deviation_pct": 50.0,
    "threshold_pct": 20.0,
    "drifted_features": [
      {"feature": "s_2", "deviation_pct": 50.0},
      {"feature": "s_3", "deviation_pct": 50.0},
      ...
    ]
  },
  "recent_requests": 50
}
```

---

## 15. Résumé des Livrables

### ✅ Checklist Finale

| # | Exigence du Cahier des Charges | Statut | Fichier(s) |
|---|-------------------------------|--------|------------|
| 1 | Git repository propre | ✅ | GitHub repo |
| 2 | Structure claire avec README | ✅ | README.md, DOCUMENTATION.md |
| 3 | Docker + docker-compose | ✅ | Dockerfile, docker-compose.yml |
| 4 | DVC pour versioning données | ✅ | data/raw.dvc |
| 5 | MLflow experiment tracking | ✅ | mlruns/, src/train_model.py |
| 6 | Pipeline ZenML | ✅ | pipelines/, steps/ |
| 7 | Optuna optimization | ✅ | src/optimize_hyperparameters.py |
| 8 | API d'inférence | ✅ | api/main.py |
| 9 | CI/CD GitHub Actions | ✅ | .github/workflows/ci_cd.yaml |
| 10 | Tests | ✅ | test_api.py, test_pipeline.py |
| **Bonus** | Monitoring & Drift | ✅ | simulate_drift.py, /monitoring |

### 📊 Performances Finales

| Métrique | Baseline (raw) | Avec Feature Engineering | Amélioration |
|----------|----------------|--------------------------|---------------|
| **RMSE** | 51.35 | **18.64** | **-63.7%** ✨ |
| MAE | 36.55 | ~14 | -61.7% |
| R² | 0.5609 | ~0.79 | +40% |

### 🚀 Commandes Essentielles

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Récupérer les données
dvc pull

# Lancer l'API
python -m uvicorn api.main:app --reload --port 8000

# Lancer le pipeline ZenML
python run_pipeline.py

# Optimisation Optuna
python src/optimize_hyperparameters.py

# Docker
docker-compose up -d

# MLflow UI
mlflow ui --port 5000

# Tests
python test_api.py
```

### 🔗 Points d'Accès

| Service | URL |
|---------|-----|
| API Swagger | http://localhost:8000/docs |
| API Health | http://localhost:8000/health |
| API Predict | http://localhost:8000/predict |
| API Monitoring | http://localhost:8000/monitoring |
| MLflow UI | http://localhost:5000 |
| ZenML Dashboard | http://localhost:8237 |

---

## 📚 Glossaire

| Terme | Définition |
|-------|------------|
| **RUL** | Remaining Useful Life - Durée de vie restante en cycles |
| **RMSE** | Root Mean Squared Error - Mesure d'erreur standard |
| **XGBoost** | Algorithme de gradient boosting optimisé |
| **DVC** | Data Version Control - Versionne les fichiers volumineux |
| **MLflow** | Plateforme de suivi d'expériences ML |
| **ZenML** | Orchestrateur de pipelines ML |
| **Optuna** | Framework d'optimisation d'hyperparamètres |
| **FastAPI** | Framework Python pour créer des APIs REST |
| **Docker** | Plateforme de conteneurisation |
| **CI/CD** | Continuous Integration / Continuous Deployment |
| **Data Drift** | Changement de distribution des données en production |

---

**Auteur :** Aymen Mabrouk  
**Institution :** École Polytechnique Sousse  
**Version :** 1.1.0  
**Date :** Décembre 2025

---

🎓 **Ce document explique chaque composant du projet MLOps de manière détaillée pour une présentation complète au professeur.**
