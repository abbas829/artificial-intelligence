
# 🚀 DEPLOYMENT GUIDE FOR CHURNGUARD AI

## Option 1: Streamlit Community Cloud (FREE - Easiest!)
### Perfect for demos, prototypes, and small teams

**Steps:**
1. Push your code to a public GitHub repository
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app" → select your repo
5. Set main file path to `churn_guard_app.py`
6. Click Deploy! 🎉

**Requirements:**
- `requirements.txt` in repo root:
```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.17.0
shap==0.42.1
joblib==1.3.2
```

**Pros:** ✅ Free, automatic HTTPS, GitHub integration, easy updates
**Cons:** ❌ Public repos only (for free tier), resource limits

---

## Option 2: Docker Container (Recommended for Production)
### Portable, consistent, scalable

**Create `Dockerfile`:**
```dockerfile
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model files
COPY churn_guard_app.py .
COPY *.pkl .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["streamlit", "run", "churn_guard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build & Run:**
```bash
# Build image
docker build -t churnguard-ai .

# Run container
docker run -p 8501:8501 churnguard-ai

# Access at http://localhost:8501
```

**Push to Docker Hub:**
```bash
docker tag churnguard-ai yourusername/churnguard-ai:latest
docker push yourusername/churnguard-ai:latest
```

**Pros:** ✅ Portable, version-controlled, works anywhere Docker runs
**Cons:** ❌ Requires Docker knowledge, manual server management

---

## Option 3: AWS Deployment (Enterprise-Grade)
### For high-availability production systems

**A) AWS Elastic Beanstalk (Easiest AWS option):**
```bash
# Install EB CLI
pip install awsebcli

# Initialize application
eb init -p docker churnguard-app

# Create environment and deploy
eb create churnguard-env
eb open
```

**B) AWS ECS + Fargate (Serverless containers):**
- Push Docker image to Amazon ECR
- Create ECS cluster with Fargate launch type
- Define task definition with your container
- Create service with Application Load Balancer
- Auto-scaling based on CPU/memory

**C) AWS EC2 (Full control):**
```bash
# Launch EC2 instance (t3.medium recommended)
# SSH into instance
sudo apt update && sudo apt install docker.io
sudo docker run -d -p 80:8501 yourusername/churnguard-ai
```

**Pros:** ✅ Enterprise-grade, auto-scaling, load balancing, monitoring
**Cons:** ❌ Complex setup, cost considerations, AWS expertise needed

---

## Option 4: Heroku (Quick PaaS)
### Platform-as-a-Service simplicity

**Create `Procfile`:**
```
web: streamlit run churn_guard_app.py --server.port=$PORT --server.address=0.0.0.0
```

**Deploy:**
```bash
# Login to Heroku
heroku login

# Create app
heroku create churnguard-ai

# Set Python buildpack
heroku buildpacks:set heroku/python

# Push and deploy
git push heroku main

# Open app
heroku open
```

**Pros:** ✅ Simple git-based deploy, free tier available, managed infrastructure
**Cons:** ❌ Free tier sleeps after inactivity, limited resources

---

## Option 5: Google Cloud Run (Serverless)
### Pay-per-use, auto-scaling to zero

**Deploy with gcloud CLI:**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/churnguard-ai

# Deploy to Cloud Run
gcloud run deploy churnguard-ai \
  --image gcr.io/your-project/churnguard-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

**Pros:** ✅ Serverless pricing, auto-scales to zero, fast cold starts, global CDN
**Cons:** ❌ Google Cloud learning curve, request timeout limits (60 min max)

---

## 🏆 DEPLOYMENT RECOMMENDATION MATRIX

| Use Case | Recommended Platform | Why |
|----------|---------------------|-----|
| Demo/Presentation | Streamlit Cloud | Free, instant, shareable URL |
| Small Team (< 50 users) | Heroku or Streamlit Cloud | Low cost, easy maintenance |
| Medium Business | Docker + AWS ECS | Scalable, professional, cost-effective |
| Enterprise | AWS ECS/Kubernetes | High availability, security, compliance |
| Variable Traffic | Google Cloud Run | Pay-per-use, auto-scaling |
| On-Premises | Docker + Internal Server | Data privacy, internal network |

---

## 🔒 SECURITY BEST PRACTICES

1. **Never commit model files to public repos** (use Git LFS or S3)
2. **Use environment variables** for sensitive config:
   ```python
   import os
   MODEL_PATH = os.getenv('MODEL_PATH', 'churn_model.pkl')
   ```
3. **Enable authentication** in Streamlit:
   ```python
   # Add to app.py
   if not st.session_state.get('authenticated'):
       password = st.text_input("Password", type="password")
       if password != os.getenv('APP_PASSWORD'):
           st.stop()
       st.session_state.authenticated = True
   ```
4. **Use HTTPS** everywhere (handled automatically by most platforms)
5. **Monitor logs** for unusual access patterns

---

## 📊 MONITORING & MAINTENANCE

After deployment, set up:
- **Uptime monitoring** (UptimeRobot, Pingdom)
- **Performance tracking** (Streamlit analytics, CloudWatch)
- **Model drift detection** (compare prediction distributions weekly)
- **User feedback collection** (add thumbs up/down in app)
- **A/B testing** (test different UI layouts for conversion)

🎉 **Your ChurnGuard AI is now ready for the world!**
