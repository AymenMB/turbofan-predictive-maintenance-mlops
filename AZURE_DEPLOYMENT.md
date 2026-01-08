# 🚀 AeroGuard AI - MLOps Deployment Pipeline

This document explains exactly how your code travels from your laptop to the **Azure cloud**. This process is fully automated using **GitHub Actions (CI/CD)**.

---

## 📊 The Workflow Visualization

```
☁️ AZURE INFRASTRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                                                              
                    ┌───────────────────────┐      ┌───────────────────────┐  
                    │  Azure Container      │      │  Azure Container      │  
                    │  Registry (ACR)       │─────▶│  Apps (ACA)           │  
                    │  ┌─────────────────┐  │      │  ┌─────────────────┐  │  
                    │  │ aeroguard-api   │  │      │  │ ✈️ aeroguard-api │  │  
                    │  │     :latest     │  │      │  │   Running...    │  │  
                    │  └─────────────────┘  │      │  └─────────────────┘  │  
                    └───────────────────────┘      └───────────┬───────────┘  
                              ▲                                │              
                              │ Push Image                     │ Serve API   
                              │                                ▼              
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                                    🌍 LIVE APPLICATION       
                                                    https://aeroguard-api...  
                              ▲                     azurecontainerapps.io     
                              │                                │              
                              │                                ▼              
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 GITHUB ACTIONS                                                             
                                                                              
    ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐  
    │ Checkout │────▶│ Login Azure  │────▶│ 🐳 Build     │────▶│ ⬆️ Push   │  
    │   Code   │     │   (Secret)   │     │ Docker Image │     │ to ACR    │  
    └──────────┘     └──────────────┘     └──────────────┘     └───────────┘  
         ▲                                                                    
         │ Trigger: push to main                                              
         │                                                                    
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💻 LOCAL DEVELOPMENT                                                          
                                                                              
    ┌──────────┐     ┌──────────────┐     ┌──────────────┐                    
    │  Write   │────▶│ git commit   │────▶│  git push    │─────────────────▶  
    │   Code   │     │ -m "message" │     │ origin main  │     GitHub Repo   
    └──────────┘     └──────────────┘     └──────────────┘                    
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🛠️ Step-by-Step Explanation

### 1. 💻 The Trigger (Local → GitHub)

Every time you run `git push`, you save your changes to the GitHub repository.

| Item | Value |
|------|-------|
| **File** | `api/main.py`, `streamlit_app.py`, or any file |
| **Action** | `git push origin main` |
| **Result** | Code is updated in the cloud repository |

```bash
# Example
git add -A
git commit -m "feat: Add new prediction endpoint"
git push origin main
```

---

### 2. 🤖 The Build Agent (GitHub Actions)

GitHub sees the new code and wakes up a **Runner** (a temporary virtual machine) to execute your instructions defined in `.github/workflows/ci_cd.yaml`.

| Step | Description |
|------|-------------|
| **A. Checkout** | The runner downloads your code |
| **B. Login** | It logs into Azure using `AZURE_CREDENTIALS` secret |
| **C. Docker Magic** | It runs `docker build`, which: |
| | - Reads your `Dockerfile` |
| | - Installs Python 3.9 |
| | - Installs libraries (xgboost, fastapi, pandas) |
| | - Copies the XGBoost model (`model_optimized.ubj`) |
| | - Creates a "Snapshot" (Docker Image) |

```yaml
# From .github/workflows/ci_cd.yaml
- name: Build Docker image
  run: |
    docker build -t aeroguard-api:latest .
```

---

### 3. ☁️ The Registry (ACR)

The runner takes that Docker Image and pushes it to **Azure Container Registry (ACR)**.

> **What is ACR?** Think of it like a private "App Store" just for your applications. It safely stores the `aeroguard-api:latest` image.

```bash
# Push command (done by GitHub Actions)
docker push acrname.azurecr.io/aeroguard-api:latest
```

---

### 4. 🚀 The Deployment (Azure Container Apps)

Finally, the runner tells **Azure Container Apps (ACA)** to update.

| Action | Description |
|--------|-------------|
| **Command** | `az containerapp update --image aeroguard-api:latest` |
| **What Happens** | Azure pulls the new image from ACR |
| | Spins up a new container (Replica) |
| | Runs a Health Check (`/health`) |
| | If healthy, switches traffic to new container |
| | Shuts down the old container |

```bash
# Update command (done by GitHub Actions)
az containerapp update \
  --name aeroguard-api \
  --resource-group mlops-rg \
  --image acrname.azurecr.io/aeroguard-api:latest
```

---

## ✅ Result

Your **AeroGuard AI API** is now running the new code **live on the internet**, accessible via the URL, without you needing to touch the server manually!

```
🌍 LIVE URL: https://aeroguard-api.salmonfield-cb3d4cec.francecentral.azurecontainerapps.io

Available Endpoints:
  GET  /health          → Health check ✅ WORKING
  POST /predict         → Single RUL prediction
  POST /predict/batch   → Batch predictions
  GET  /model-info      → Model information
  GET  /docs            → Swagger API documentation ✅ WORKING
```

---

## 📁 Key Files in This Pipeline

| File | Purpose |
|------|---------|
| `Dockerfile` | Instructions to build the container |
| `.github/workflows/ci_cd.yaml` | CI/CD pipeline definition |
| `model_optimized.ubj` | Trained XGBoost model (RMSE: 18.64) |
| `api/main.py` | FastAPI application |
| `feature_columns.txt` | Expected feature names for model |

---

## 🔐 Required GitHub Secrets

Before deploying, add these secrets to your GitHub repository:

| Secret Name | Description |
|-------------|-------------|
| `AZURE_CREDENTIALS` | Service principal credentials (JSON) |
| `ACR_LOGIN_SERVER` | e.g., `acrname.azurecr.io` |
| `ACR_USERNAME` | Registry username |
| `ACR_PASSWORD` | Registry password |

```bash
# How to get Azure credentials
az ad sp create-for-rbac --name "aeroguard-github" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/mlops-rg \
  --sdk-auth
```

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 18.64 cycles |
| **R²** | 0.79 |
| **Improvement** | 63.7% from baseline |
| **Dataset** | NASA C-MAPSS FD001 |

---

## 🚁 Quick Local Test

Before deploying, test locally:

```bash
# Build Docker image
docker build -t aeroguard-api .

# Run container
docker run -p 8000:8000 aeroguard-api

# Test endpoint
curl http://localhost:8000/health
```

---

## 📚 References

- [GitHub Repository](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops)
- [Azure Container Apps Documentation](https://docs.microsoft.com/azure/container-apps/)
- [NASA C-MAPSS Dataset](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)
