# api.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
import pickle
import numpy as np
import tempfile
import os
import uvicorn
from datetime import datetime
import logging

# Import your existing modules
from enhanced_video_analyzer import VideoAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advertisement Success Predictor API",
    description="API for predicting advertisement success using ML and computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
@app.on_event("startup")
async def load_models():
    global rating_model, success_model, money_model, feature_names, categorical_cols, numerical_cols
    try:
        with open("model/model.pkl", "rb") as f:
            models = pickle.load(f)
        
        rating_model = models["rating_model"]
        success_model = models["success_model"]
        money_model = models["money_model"]
        feature_names = models.get("feature_names", [])
        categorical_cols = models.get("categorical_cols", [])
        numerical_cols = models.get("numerical_cols", [])
        
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        rating_model = success_model = money_model = None

# Initialize video analyzer
video_analyzer = VideoAnalyzer()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    relationship_status: str = Field(..., description="Relationship status of target audience")
    industry: str = Field(..., description="Industry of the advertisement")
    genre: str = Field(..., description="Genre of the advertisement")
    targeted_sex: str = Field(..., description="Targeted sex (Male/Female/All)")
    average_runtime: float = Field(..., description="Average runtime in minutes per week")
    airtime: str = Field(..., description="Airtime slot (Morning/Daytime/Primetime)")
    airlocation: str = Field(..., description="Location where ad will air")
    expensive: str = Field(..., description="Expensive level (Low/Medium/High)")
    
    class Config:
        schema_extra = {
            "example": {
                "relationship_status": "Married-civ-spouse",
                "industry": "Pharma",
                "genre": "Comedy",
                "targeted_sex": "Male",
                "average_runtime": 40,
                "airtime": "Primetime",
                "airlocation": "United-States",
                "expensive": "Low"
            }
        }

class PredictionResponse(BaseModel):
    predicted_rating: float
    success_status: int
    success_probability: float
    money_back_guarantee: str
    model_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)

class VideoAnalysisResponse(PredictionResponse):
    video_analysis: Dict[str, Any]
    video_report: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=rating_model is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        models_loaded=rating_model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction based on advertisement parameters
    """
    if rating_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert request to DataFrame
        input_data = {
            'relationship_status': request.relationship_status,
            'industry': request.industry,
            'genre': request.genre,
            'targeted_sex': request.targeted_sex,
            'average_runtime(minutes_per_week)': request.average_runtime,
            'airtime': request.airtime,
            'airlocation': request.airlocation,
            'expensive': request.expensive
        }
        
        features = pd.DataFrame([input_data])
        
        # Make predictions
        rating = rating_model.predict(features)[0]
        pred = success_model.predict(features)[0]
        prob = success_model.predict_proba(features)[0][1] * 100
        money_pred = money_model.predict(features)[0]
        
        return PredictionResponse(
            predicted_rating=float(rating),
            success_status=int(pred),
            success_probability=float(prob),
            money_back_guarantee=str(money_pred)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-with-video", response_model=VideoAnalysisResponse)
async def predict_with_video(
    relationship_status: str = Form(...),
    industry: str = Form(...),
    genre: str = Form(...),
    targeted_sex: str = Form(...),
    average_runtime: float = Form(...),
    airtime: str = Form(...),
    airlocation: str = Form(...),
    expensive: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Make prediction with video analysis
    """
    if rating_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    video_path = None
    try:
        # Validate video file
        allowed_types = ["video/mp4", "video/mov", "video/avi", "video/mkv", 
                        "video/x-msvideo", "video/quicktime"]
        if video.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_types}"
            )
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            if len(content) > 100 * 1024 * 1024:  # 100MB limit
                raise HTTPException(status_code=400, detail="File too large (max 100MB)")
            
            tmp.write(content)
            video_path = tmp.name
        
        # Make ML prediction
        input_data = {
            'relationship_status': relationship_status,
            'industry': industry,
            'genre': genre,
            'targeted_sex': targeted_sex,
            'average_runtime(minutes_per_week)': average_runtime,
            'airtime': airtime,
            'airlocation': airlocation,
            'expensive': expensive
        }
        
        features = pd.DataFrame([input_data])
        
        rating = rating_model.predict(features)[0]
        pred = success_model.predict(features)[0]
        prob = success_model.predict_proba(features)[0][1] * 100
        money_pred = money_model.predict(features)[0]
        
        # Analyze video
        report = video_analyzer.analyze_ad_video(
            video_path=video_path,
            ml_rating=float(rating),
            ml_success_prob=float(prob),
            ml_money_pred=str(money_pred)
        )
        
        # Extract metrics from report (simplified)
        video_metrics = {
            "duration": "Calculated from video",
            "brightness": "Analyzed",
            "contrast": "Analyzed",
            "motion": "Analyzed",
            "face_detection": "Performed",
            "scene_cuts": "Detected"
        }
        
        return VideoAnalysisResponse(
            predicted_rating=float(rating),
            success_status=int(pred),
            success_probability=float(prob),
            money_back_guarantee=str(money_pred),
            video_analysis=video_metrics,
            video_report=report
        )
        
    except Exception as e:
        logger.error(f"Prediction with video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception as e:
                logger.error(f"Error deleting temp file: {e}")

@app.get("/model-info")
async def model_info():
    """
    Get information about loaded models
    """
    if rating_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models_loaded": True,
        "features": feature_names,
        "categorical_features": categorical_cols,
        "numerical_features": numerical_cols,
        "model_type": {
            "rating": "RandomForestRegressor",
            "success": "GradientBoostingClassifier",
            "money": "RandomForestClassifier"
        }
    }

@app.post("/analyze-video-only")
async def analyze_video_only(
    video: UploadFile = File(...),
    ml_rating: Optional[float] = Form(3.0),
    ml_prob: Optional[float] = Form(50.0),
    ml_money: Optional[str] = Form("Unknown")
):
    """
    Analyze video without ML prediction
    """
    video_path = None
    try:
        # Validate video file
        allowed_types = ["video/mp4", "video/mov", "video/avi", "video/mkv"]
        if video.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            if len(content) > 100 * 1024 * 1024:  # 100MB limit
                raise HTTPException(status_code=400, detail="File too large (max 100MB)")
            
            tmp.write(content)
            video_path = tmp.name
        
        # Analyze video
        report = video_analyzer.analyze_ad_video(
            video_path=video_path,
            ml_rating=ml_rating,
            ml_success_prob=ml_prob,
            ml_money_pred=ml_money
        )
        
        return {
            "video_analysis_report": report,
            "video_name": video.filename,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception as e:
                logger.error(f"Error deleting temp file: {e}")

@app.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from the rating model
    """
    if rating_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Extract feature importance from Random Forest
        rf_model = rating_model.named_steps['regressor']
        importance = rf_model.feature_importances_
        
        # Get feature names (after preprocessing)
        feature_names_all = categorical_cols + numerical_cols
        
        # Create importance dict
        importance_dict = {
            name: float(imp) 
            for name, imp in zip(feature_names_all[:len(importance)], importance)
        }
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:5]
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )