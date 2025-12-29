# CI/CD Pipeline Guide - GitHub Actions

## Overview

This CI/CD pipeline automates testing, building, and validation of the Turbofan RUL MLOps project on every push to the main branch.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions Trigger                    │
│              (push/pull_request to main branch)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│  Test & Lint  │           │   Security    │
│   (Job 1)     │           │     Scan      │
└───────┬───────┘           └───────────────┘
        │
        ├──────────────┬──────────────┐
        │              │              │
        ▼              ▼              ▼
┌───────────────┐ ┌─────────────┐ ┌─────────────┐
│ Build Docker  │ │  ML Pipeline│ │  Security   │
│   (Job 2)     │ │    Test     │ │    Scan     │
│               │ │  (Job 3)    │ │  (Job 4)    │
└───────┬───────┘ └──────┬──────┘ └─────────────┘
        │                │
        └────────┬───────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Deployment    │
        │    Summary      │
        │    (Job 5)      │
        └─────────────────┘
```

## Jobs Breakdown

### Job 1: Test & Lint ✅

**Purpose:** Validate code quality and run tests

**Steps:**
1. **Checkout code** - Get repository contents
2. **Setup Python 3.9** - Install Python environment
3. **Install dependencies** - Install all required packages
4. **Lint with flake8** - Check code quality
   - Strict errors: `E9,F63,F7,F82` (syntax errors, undefined names)
   - Warnings: max-complexity=10, max-line-length=120
5. **Check formatting** - Black code formatter (informational)
6. **Run unit tests** - Execute pytest on test_api.py
7. **Upload test results** - Save coverage reports

**Exit Criteria:**
- ✅ No syntax errors
- ✅ All tests pass
- ⚠️ Warnings allowed (non-blocking)

---

### Job 2: Build Docker Container ✅

**Purpose:** Build and validate Docker image

**Dependencies:** Requires Job 1 (Test & Lint) to pass

**Steps:**
1. **Checkout code** - Get repository contents
2. **Setup Docker Buildx** - Configure Docker builder
3. **Check Dockerfile** - Verify Dockerfile exists
4. **Build Docker image** - Build `turbofan-rul-api:latest`
5. **Smoke test** - Start container and test health endpoint
   ```bash
   docker run -d --name test-api -p 8000:8000 turbofan-rul-api:latest
   curl -f http://localhost:8000/health
   ```
6. **Save artifact** - Export Docker image as tar.gz
7. **Upload artifact** - Store image for 7 days

**Exit Criteria:**
- ✅ Docker image builds successfully
- ✅ Container starts without errors
- ✅ Health endpoint responds with 200 OK

**Optional:** Push to Docker Hub (commented out, can be enabled with secrets)

---

### Job 3: ML Pipeline Simulation ✅

**Purpose:** Validate ML training logic and pipeline structure

**Dependencies:** Requires Job 1 (Test & Lint) to pass

**Steps:**
1. **Checkout code** - Get repository contents with full history
2. **Setup Python 3.9** - Install Python environment
3. **Install dependencies** - Install all required packages
4. **Setup DVC** - Configure data version control
5. **Download data** - Pull data via DVC or use cached
6. **Test preprocessing** - Validate data_preprocessing module
7. **Test training** - Validate XGBoost and model loading
8. **Validate artifacts** - Check for model files

**Note:** Full ZenML pipeline execution requires:
- ZenML Cloud or server connection
- DVC remote storage
- MLflow tracking server
- Training data access

For CI/CD demo, this job validates:
- ✅ Code imports work
- ✅ Module structure is correct
- ✅ Dependencies are satisfied

**Exit Criteria:**
- ✅ All modules importable
- ✅ Model artifacts validated (if present)
- ⚠️ Full pipeline execution optional (requires infrastructure)

---

### Job 4: Security Scan 🔒

**Purpose:** Scan for vulnerabilities

**Dependencies:** Requires Job 1 (Test & Lint) to pass

**Steps:**
1. **Trivy scanner** - Scan filesystem for vulnerabilities
2. **Upload SARIF** - Send results to GitHub Security tab
3. **Python safety check** - Check dependencies for known CVEs

**Exit Criteria:**
- ⚠️ Informational only (non-blocking)
- Results available in GitHub Security tab

---

### Job 5: Deployment Summary 📊

**Purpose:** Generate execution report

**Dependencies:** Requires all previous jobs to complete

**Steps:**
1. **Generate report** - Show pipeline status
2. **Display results** - Show pass/fail for each job
3. **Notify on failure** - Exit with error if any job failed

**Output:**
```
========================================
🚀 CI/CD Pipeline Execution Summary
========================================

Pipeline run: 42
Commit: abc123def456
Branch: main
Triggered by: push

Job Results:
  ✅ Test & Lint: success
  ✅ Build Container: success
  ✅ ML Pipeline Simulation: success

========================================
📊 Deliverables Status
========================================
✅ Code Quality: Passed
✅ Unit Tests: Passed
✅ Docker Image: Built & Tested
✅ ML Pipeline: Validated

🎯 Ready for deployment!
========================================
```

---

## How to Use

### 1. Initial Setup (One-time)

The workflow file is already created at:
```
.github/workflows/ci_cd.yaml
```

### 2. Commit and Push

```bash
# Navigate to project directory
cd "D:\cycleing\5eme\R\mlops projet"

# Check status
git status

# Add workflow file
git add .github/workflows/ci_cd.yaml

# Add any other changes
git add .

# Commit
git commit -m "Add GitHub Actions CI/CD pipeline with Docker build and ML validation"

# Push to GitHub
git push origin main
```

### 3. Monitor Pipeline Execution

1. **Go to GitHub repository:**
   ```
   https://github.com/AymenMB/turbofan-predictive-maintenance-mlops
   ```

2. **Click "Actions" tab**

3. **View running workflow:**
   - Click on the latest workflow run
   - See real-time logs for each job
   - Check job status (✅ or ❌)

4. **Inspect artifacts:**
   - Docker image (tar.gz)
   - Test results
   - Coverage reports

---

## Pipeline Triggers

The pipeline runs automatically on:

1. **Push to main branch:**
   ```bash
   git push origin main
   ```

2. **Pull request to main:**
   ```bash
   git checkout -b feature/my-feature
   git push origin feature/my-feature
   # Create PR on GitHub
   ```

3. **Manual trigger (via GitHub UI):**
   - Go to Actions tab
   - Select workflow
   - Click "Run workflow"

---

## Environment Variables

Defined in the workflow:

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHON_VERSION` | 3.9 | Python version for CI |
| `DOCKER_IMAGE_NAME` | turbofan-rul-api | Docker image name |

---

## Secrets (Optional)

For Docker Hub push (currently commented out):

1. **Add secrets in GitHub:**
   - Go to Settings → Secrets and variables → Actions
   - Add `DOCKER_USERNAME`
   - Add `DOCKER_PASSWORD`

2. **Uncomment Docker push steps** in workflow file

---

## Artifacts

Pipeline generates these artifacts:

1. **Test Results** (30 days retention)
   - Coverage reports
   - Test logs

2. **Docker Image** (7 days retention)
   - turbofan-api-image.tar.gz
   - ~500MB compressed

3. **Security Scan Results**
   - Available in GitHub Security tab
   - SARIF format

---

## Troubleshooting

### Issue 1: Test Fails

**Problem:** pytest fails in CI

**Solution:**
```bash
# Test locally first
.\.venv\Scripts\python.exe -m pytest test_api.py -v

# Check if test_api.py exists
ls test_api.py

# Install missing dependencies
pip install pytest requests
```

### Issue 2: Docker Build Fails

**Problem:** Docker image build fails

**Solution:**
```bash
# Test Docker build locally
docker build -t turbofan-rul-api:latest .

# Check Dockerfile exists
ls Dockerfile

# Review Docker logs in GitHub Actions
```

### Issue 3: Linting Errors

**Problem:** flake8 reports errors

**Solution:**
```bash
# Run flake8 locally
flake8 src api --count --select=E9,F63,F7,F82 --show-source --statistics

# Auto-fix with black
pip install black
black src api

# Commit fixes
git add .
git commit -m "Fix linting issues"
git push origin main
```

### Issue 4: ML Pipeline Fails

**Problem:** ZenML or model tests fail

**Solution:**
- Check if model_optimized.ubj exists in repo (optional)
- Verify all Python packages installed
- Check import statements in source files
- Review logs in GitHub Actions

---

## Performance

| Job | Duration | Resources |
|-----|----------|-----------|
| Test & Lint | ~2-3 min | 2 CPU, 7GB RAM |
| Build Docker | ~3-5 min | 2 CPU, 7GB RAM |
| ML Pipeline | ~1-2 min | 2 CPU, 7GB RAM |
| Security Scan | ~2-3 min | 2 CPU, 7GB RAM |
| **Total** | **~8-13 min** | ubuntu-latest |

---

## Cost

**GitHub Actions Free Tier:**
- 2,000 minutes/month for private repos
- Unlimited for public repos

**This pipeline uses:**
- ~10 minutes per run
- ~200 runs/month = 2,000 minutes (within free tier)

---

## Production Enhancements

### 1. Add Docker Registry Push

Uncomment and configure Docker Hub push:
```yaml
- name: Login to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKER_USERNAME }}
    password: ${{ secrets.DOCKER_PASSWORD }}

- name: Push to Docker Hub
  run: |
    docker push username/turbofan-rul-api:latest
    docker push username/turbofan-rul-api:${{ github.sha }}
```

### 2. Add Deployment Stage

Deploy to cloud platform:
```yaml
deploy:
  name: Deploy to Production
  runs-on: ubuntu-latest
  needs: [build-container]
  if: github.ref == 'refs/heads/main'
  steps:
    - name: Deploy to AWS/Azure/GCP
      run: |
        # Deploy Docker image to cloud
```

### 3. Add Notification

Send notifications on success/failure:
```yaml
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 4. Add Performance Testing

Load test the API:
```yaml
- name: Load test
  run: |
    pip install locust
    locust -f locustfile.py --headless -u 10 -r 2 --run-time 1m
```

---

## Best Practices

1. **✅ Keep workflows fast** - Cache dependencies, use matrix builds
2. **✅ Fail fast** - Stop on critical errors
3. **✅ Use artifacts** - Save build outputs for debugging
4. **✅ Security scans** - Regularly scan for vulnerabilities
5. **✅ Separate concerns** - One job per responsibility
6. **✅ Document** - Clear job names and comments

---

## Workflow Status Badge

Add to README.md:
```markdown
![CI/CD Pipeline](https://github.com/AymenMB/turbofan-predictive-maintenance-mlops/workflows/CI%2FCD%20Pipeline%20-%20Turbofan%20RUL%20MLOps/badge.svg)
```

This shows build status in your README.

---

## Summary

### ✅ What This Pipeline Does

1. **Validates code quality** (flake8, black)
2. **Runs unit tests** (pytest)
3. **Builds Docker image** (validates Dockerfile)
4. **Tests Docker container** (smoke test)
5. **Validates ML pipeline** (imports, structure)
6. **Scans for security issues** (Trivy, safety)
7. **Generates execution report** (summary)

### ✅ What This Ensures

- Code quality maintained
- Tests pass before merge
- Docker image builds successfully
- API container works
- ML code is importable
- Security vulnerabilities detected

### 🚀 Next Steps

1. ✅ Commit and push workflow file
2. ✅ Monitor first pipeline run on GitHub
3. ✅ Fix any issues that arise
4. ✅ Add status badge to README
5. ⏳ Configure Docker Hub push (optional)
6. ⏳ Add deployment stage (optional)

---

**Pipeline Status:** Ready to deploy! 🎯

Push to GitHub and watch it run automatically! 🚀
