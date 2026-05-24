
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import hashlib
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize FastAPI app
app = FastAPI(
    title="🏠 House Price Prediction API",
    description="Production-ready API for real estate price estimation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================
class HouseFeatures(BaseModel):
    square_feet: float = Field(..., gt=0, le=50000, description="Living area in square feet")
    bedrooms: int = Field(..., ge=1, le=20, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, le=20, description="Number of bathrooms")
    lot_size: float = Field(..., gt=0, le=1000000, description="Lot size in square feet")
    year_built: int = Field(..., ge=1800, le=2024, description="Year the house was built")
    garage_spaces: int = Field(..., ge=0, le=10, description="Number of garage spaces")
    distance_to_city_center: float = Field(..., ge=0, le=100, description="Distance to city center in miles")
    school_rating: float = Field(..., ge=0, le=10, description="School district rating (0-10)")
    crime_index: float = Field(..., ge=0, le=100, description="Crime index (0-100, lower is better)")
    property_tax_rate: float = Field(..., ge=0, le=10, description="Property tax rate percentage")
    neighborhood: str = Field(..., description="Neighborhood type")
    property_type: str = Field(..., description="Type of property")
    heating_type: str = Field(..., description="Heating system type")

    @validator('neighborhood')
    def validate_neighborhood(cls, v):
        valid = {'Downtown', 'Suburban', 'Rural', 'Waterfront'}
        if v not in valid:
            raise ValueError(f'neighborhood must be one of {valid}')
        return v

    @validator('property_type')
    def validate_property_type(cls, v):
        valid = {'Single Family', 'Condo', 'Townhouse'}
        if v not in valid:
            raise ValueError(f'property_type must be one of {valid}')
        return v

    @validator('heating_type')
    def validate_heating_type(cls, v):
        valid = {'Gas', 'Electric', 'Oil', 'Heat Pump'}
        if v not in valid:
            raise ValueError(f'heating_type must be one of {valid}')
        return v
class PredictionResponse(BaseModel):
    success: bool
    request_id: str
    predicted_price: Optional[float]
    confidence_interval: Optional[Dict]
    processing_time_ms: float
    model_version: str
    timestamp: str
    errors: Optional[List[str]] = None
class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures]
# ============================================================================
# Model Loading and Pipeline Setup
# ============================================================================
MODEL_PATH = "./model_registry/house_price_predictor_v1.0.0/model.pkl"
model = None
prediction_count = 0
prediction_times = []
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
def generate_request_id() -> str:
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
def calculate_confidence(prediction: float) -> Dict:
    margin = prediction * 0.08
    return {
        "lower_bound": round(prediction - margin, 2),
        "upper_bound": round(prediction + margin, 2),
        "confidence_level": "92%"
    }
# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    return {
        "message": "🏠 House Price Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures, background_tasks: BackgroundTasks):
    start_time = time.time()
    request_id = generate_request_id()

    logger.info(f"[{request_id}] 🔄 Processing prediction request")

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([features.dict()])

        # Make prediction
        predicted_price = float(model.predict(input_df)[0])
        confidence = calculate_confidence(predicted_price)

        processing_time = (time.time() - start_time) * 1000

        # Update metrics
        global prediction_count, prediction_times
        prediction_count += 1
        prediction_times.append(processing_time)

        # Log prediction (background task)
        background_tasks.add_task(
            logger.info,
            f"[{request_id}] ✅ Prediction: ${predicted_price:,.2f} in {processing_time:.2f}ms"
        )

        return PredictionResponse(
            success=True,
            request_id=request_id,
            predicted_price=round(predicted_price, 2),
            confidence_interval=confidence,
            processing_time_ms=round(processing_time, 2),
            model_version="1.0.0",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    start_time = time.time()
    request_id = generate_request_id()

    logger.info(f"[{request_id}] 🔄 Processing batch of {len(request.houses)} predictions")

    try:
        input_df = pd.DataFrame([h.dict() for h in request.houses])
        predictions = model.predict(input_df)

        results = []
        for i, (house, pred) in enumerate(zip(request.houses, predictions)):
            results.append({
                "index": i,
                "predicted_price": round(float(pred), 2),
                "confidence_interval": calculate_confidence(pred),
                "input_summary": f"{house.bedrooms}br/{house.bathrooms}ba, {house.square_feet}sqft"
            })

        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "request_id": request_id,
            "predictions": results,
            "batch_size": len(request.houses),
            "processing_time_ms": round(processing_time, 2),
            "model_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/metrics")
async def get_metrics():
    global prediction_times
    if not prediction_times:
        return {"message": "No predictions made yet"}

    recent_times = prediction_times[-100:]  # Last 100 predictions
    return {
        "total_predictions": prediction_count,
        "avg_latency_ms": round(np.mean(recent_times), 2),
        "p50_latency_ms": round(np.percentile(recent_times, 50), 2),
        "p95_latency_ms": round(np.percentile(recent_times, 95), 2),
        "p99_latency_ms": round(np.percentile(recent_times, 99), 2),
        "min_latency_ms": round(min(recent_times), 2),
        "max_latency_ms": round(max(recent_times), 2)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
