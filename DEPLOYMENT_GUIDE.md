# FastAPI Deployment & Docker Guide

## Overview

This guide explains how to deploy the optimized Turbofan RUL prediction model as a REST API using FastAPI and Docker.

## API Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  POST        │    │  GET         │    │  GET         │  │
│  │  /predict    │    │  /health     │    │  /model-info │  │
│  └──────┬───────┘    └──────────────┘    └──────────────┘  │
│         │                                                    │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           XGBoost Model (model_optimized.ubj)        │   │
│  │           RMSE: 50.71 cycles                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Files Created

```
project/
├── api/
│   ├── __init__.py           # API package initializer
│   └── main.py               # FastAPI application
├── Dockerfile                # Container definition
├── docker-compose.yml        # Docker Compose configuration
├── .dockerignore            # Docker build exclusions
└── test_api.py              # API testing script
```

## API Endpoints

### 1. `POST /predict`
Predict Remaining Useful Life (RUL) for a turbofan engine.

**Request Body**:
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

**Response**:
```json
{
  "RUL": 112.45,
  "status": "Healthy",
  "confidence": "High"
}
```

**Status Classification**:
- `Critical`: RUL < 30 cycles (immediate maintenance required)
- `Warning`: RUL < 80 cycles (schedule maintenance soon)
- `Healthy`: RUL ≥ 80 cycles (normal operation)

### 2. `GET /health`
Check API health status.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "model_optimized.ubj"
}
```

### 3. `GET /model-info`
Get information about the loaded model.

**Response**:
```json
{
  "model_type": "XGBoost Booster",
  "optimization": "Optuna (20 trials)",
  "performance": {
    "test_rmse": 50.71,
    "improvement_over_baseline": "1.26%"
  },
  "hyperparameters": {
    "learning_rate": 0.046,
    "max_depth": 3,
    "n_estimators": 287
  }
}
```

### 4. `GET /` (Root)
API information and available endpoints.

### 5. `GET /docs`
Interactive API documentation (Swagger UI).

## Testing Locally (Without Docker)

### Step 1: Install Dependencies
```bash
# Make sure you're in the virtual environment
cd "D:\cycleing\5eme\R\mlops projet"
.venv\Scripts\activate
```

### Step 2: Start the API
```bash
# Option 1: Using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python module
python -m uvicorn api.main:app --reload
```

### Step 3: Test the API
Open a new terminal and run:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run test script
python test_api.py
```

### Step 4: Access Interactive Docs
Open your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 5: Test with curl (Windows PowerShell)
```powershell
# Health check
curl http://localhost:8000/health

# Prediction
$body = @{
    setting_1 = -0.0007
    setting_2 = -0.0004
    setting_3 = 100.0
    s_1 = 518.67
    s_2 = 641.82
    s_3 = 1589.70
    s_4 = 1400.60
    s_5 = 14.62
    s_6 = 21.61
    s_7 = 554.36
    s_8 = 2388.06
    s_9 = 9046.19
    s_10 = 1.30
    s_11 = 47.47
    s_12 = 521.66
    s_13 = 2388.02
    s_14 = 8138.62
    s_15 = 8.4195
    s_16 = 0.03
    s_17 = 392
    s_18 = 2388
    s_19 = 100.0
    s_20 = 39.06
    s_21 = 23.4190
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
```

## Docker Deployment

### Prerequisites
**⚠️ MANUAL ACTION REQUIRED**: Install Docker Desktop if not already installed.

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop
3. Verify installation:
   ```bash
   docker --version
   docker-compose --version
   ```

### Step 1: Build the Docker Image
```bash
cd "D:\cycleing\5eme\R\mlops projet"
docker build -t turbofan-rul-api:latest .
```

**Expected output**:
```
[+] Building 45.2s (14/14) FINISHED
 => [internal] load build definition
 => => transferring dockerfile: 32B
 ...
 => exporting to image
 => => naming to docker.io/library/turbofan-rul-api:latest
```

### Step 2: Run with Docker Compose (Recommended)
```bash
docker-compose up -d
```

**Expected output**:
```
Creating network "turbofan-network" with the default driver
Creating turbofan-rul-api ... done
```

### Step 3: Check Container Status
```bash
docker-compose ps
```

**Expected output**:
```
       Name                     Command               State           Ports
--------------------------------------------------------------------------------
turbofan-rul-api   uvicorn api.main:app --hos ...   Up      0.0.0.0:8000->8000/tcp
```

### Step 4: View Logs
```bash
docker-compose logs -f
```

### Step 5: Test the Dockerized API
```bash
# Health check
curl http://localhost:8000/health

# Run test script
python test_api.py
```

### Step 6: Stop the Container
```bash
docker-compose down
```

## Alternative: Run Docker Image Directly

If you don't want to use docker-compose:

```bash
# Build
docker build -t turbofan-rul-api:latest .

# Run
docker run -d -p 8000:8000 --name turbofan-rul-api turbofan-rul-api:latest

# Stop
docker stop turbofan-rul-api
docker rm turbofan-rul-api
```

## Production Deployment Considerations

### 1. Environment Variables
For production, use environment variables:

```yaml
# docker-compose.prod.yml
services:
  api:
    environment:
      - MODEL_PATH=/app/models/model_optimized.ubj
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
```

### 2. Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name api.turbofan-rul.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. HTTPS with Let's Encrypt
```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d api.turbofan-rul.com
```

### 4. Monitoring
Add health checks and monitoring:

```yaml
# Add to docker-compose.yml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Troubleshooting

### Issue 1: Port Already in Use
```bash
# Windows: Find process using port 8000
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <PID> /F
```

### Issue 2: Model Not Found
Make sure `model_optimized.ubj` exists in the project root:
```bash
ls model_optimized.ubj
```

If missing, run the optimization again:
```bash
.venv\Scripts\python.exe src\optimize_hyperparameters.py
```

### Issue 3: Docker Build Fails
Clear Docker cache:
```bash
docker system prune -a
docker-compose build --no-cache
```

### Issue 4: Cannot Connect to Docker
Make sure Docker Desktop is running:
- Windows: Check system tray for Docker icon
- If not running: Start Docker Desktop application

## Performance Testing

### Load Testing with Apache Bench (Optional)
```bash
# Install Apache Bench (ab)
# Windows: https://www.apachelounge.com/download/

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -T 'application/json' -p request.json http://localhost:8000/predict
```

### Stress Testing with Locust (Optional)
```python
# locustfile.py
from locust import HttpUser, task, between

class TurbofanUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        payload = {
            "setting_1": -0.0007,
            # ... all other features
        }
        self.client.post("/predict", json=payload)
```

Run:
```bash
pip install locust
locust -f locustfile.py
# Open http://localhost:8089
```

## CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy API

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t turbofan-rul-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push turbofan-rul-api:${{ github.sha }}
```

## Summary

### Files Created
- ✅ `api/main.py` - FastAPI application (280 lines)
- ✅ `api/__init__.py` - Package initializer
- ✅ `Dockerfile` - Container definition
- ✅ `docker-compose.yml` - Docker Compose config
- ✅ `.dockerignore` - Build optimization
- ✅ `test_api.py` - API testing script

### Next Steps
1. **Test Locally**: Run `uvicorn api.main:app --reload` and `python test_api.py`
2. **Install Docker**: Download and install Docker Desktop (MANUAL ACTION REQUIRED)
3. **Build Container**: Run `docker build -t turbofan-rul-api .`
4. **Deploy**: Run `docker-compose up -d`
5. **Verify**: Access http://localhost:8000/docs

### API Features
- ✅ RUL prediction with optimized model
- ✅ Health status classification (Critical/Warning/Healthy)
- ✅ Automatic sensor preprocessing
- ✅ Interactive documentation (Swagger UI)
- ✅ Health check endpoint
- ✅ Model information endpoint
- ✅ Docker containerization
- ✅ Production-ready error handling

**Status**: Ready for deployment! 🚀
