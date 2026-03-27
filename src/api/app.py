from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import sys
sys.path.append('.')
from src.config import config

# Initialize FastAPI
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict student grades based on academic and demographic factors",
    version="1.0.0"
)

print(f"Starting API in {config.ENVIRONMENT} mode on port {config.PORT}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if config.CORS_ORIGINS != ['*'] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessor
MODEL_PATH = config.MODEL_PATH
PREPROCESSOR_PATH = config.PREPROCESSOR_PATH

model = None
preprocessor = None

def load_artifacts():
    """Load model and preprocessor"""
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✅ Model and preprocessor loaded successfully")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# Load artifacts on startup
@app.on_event("startup")
async def startup_event():
    load_artifacts()

# Pydantic models for request/response
class StudentFeatures(BaseModel):
    age: int = Field(..., ge=14, le=19, description="Student age (14-19)")
    gender: str = Field(..., pattern="^(male|female|other)$", description="Gender")
    school_type: str = Field(..., pattern="^(public|private)$", description="School type")
    parent_education: str = Field(..., pattern="^(no formal|high school|diploma|graduate|post graduate|phd)$", 
                                   description="Parent education level")
    study_hours: float = Field(..., ge=0, le=24, description="Daily study hours")
    attendance_percentage: float = Field(..., ge=0, le=100, description="Attendance percentage")
    internet_access: str = Field(..., pattern="^(yes|no)$", description="Internet access at home")
    travel_time: str = Field(..., pattern="^(<15 min|15-30 min|30-60 min|>60 min)$", 
                             description="Travel time to school")
    extra_activities: str = Field(..., pattern="^(yes|no)$", description="Participates in extra activities")
    study_method: str = Field(..., pattern="^(notes|textbook|group study|coaching|online videos|mixed)$",
                              description="Preferred study method")
    math_score: float = Field(..., ge=0, le=100, description="Mathematics score (0-100)")
    science_score: float = Field(..., ge=0, le=100, description="Science score (0-100)")
    english_score: float = Field(..., ge=0, le=100, description="English score (0-100)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 16,
                "gender": "female",
                "school_type": "private",
                "parent_education": "graduate",
                "study_hours": 5.5,
                "attendance_percentage": 85.0,
                "internet_access": "yes",
                "travel_time": "15-30 min",
                "extra_activities": "yes",
                "study_method": "notes",
                "math_score": 75.0,
                "science_score": 80.0,
                "english_score": 85.0
            }
        }

class PredictionResponse(BaseModel):
    student_id: Optional[int] = None
    predicted_grade: str
    predicted_grade_numeric: int
    confidence_scores: Dict[str, float]
    risk_level: str
    recommendation: str
    prediction_timestamp: str

class BatchPredictionRequest(BaseModel):
    students: List[StudentFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    summary: Dict[str, int]

# Helper functions
def create_features(df):
    """Create engineered features"""
    df['average_score'] = (df['math_score'] + df['science_score'] + df['english_score']) / 3
    df['total_score'] = df['math_score'] + df['science_score'] + df['english_score']
    df['study_efficiency'] = df['average_score'] / (df['study_hours'] + 1)
    df['attendance_score'] = df['attendance_percentage'] / 100
    return df

def get_risk_level(grade):
    """Get risk level based on predicted grade"""
    risk_map = {
        'f': ('High Risk', 'Immediate intervention required - critical academic support needed'),
        'e': ('High Risk', 'Urgent intervention needed - consider tutoring and counseling'),
        'd': ('Medium Risk', 'Needs improvement - regular monitoring and support recommended'),
        'c': ('Low Risk', 'Satisfactory performance - continue current efforts'),
        'b': ('Very Low Risk', 'Good performance - maintain good study habits'),
        'a': ('Very Low Risk', 'Outstanding performance - consider advanced programs')
    }
    return risk_map.get(grade, ('Unknown', ''))

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Student Performance Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": ["/predict", "/predict_batch", "/health", "/model_info"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features_count": len(preprocessor['categorical_cols'] + preprocessor['numerical_cols'] + 
                              ['average_score', 'total_score', 'study_efficiency', 'attendance_score']),
        "classes": preprocessor['label_encoders']['final_grade'].classes_.tolist(),
        "classes_count": len(preprocessor['label_encoders']['final_grade'].classes_)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentFeatures):
    """Predict grade for a single student"""
    try:
        if model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([student.model_dump()])
        
        # Create features
        df = create_features(df)
        
        # Encode categorical variables
        for col in preprocessor['categorical_cols']:
            if col in df.columns:
                try:
                    df[col] = preprocessor['label_encoders'][col].transform(df[col].astype(str))
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {e}")
        
        # Prepare features
        feature_cols = preprocessor['categorical_cols'] + preprocessor['numerical_cols'] + \
                       ['average_score', 'total_score', 'study_efficiency', 'attendance_score']
        
        X = preprocessor['scaler'].transform(df[feature_cols])
        
        # Predict
        pred_class = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        
        # Get grade names
        grade_names = preprocessor['label_encoders']['final_grade'].classes_
        predicted_grade = grade_names[pred_class]
        
        # Get risk level and recommendation
        risk_level, recommendation = get_risk_level(predicted_grade)
        
        # Confidence scores
        confidence_scores = {grade_names[i].upper(): float(pred_proba[i]) for i in range(len(grade_names))}
        
        return PredictionResponse(
            predicted_grade=predicted_grade.upper(),
            predicted_grade_numeric=int(pred_class),
            confidence_scores=confidence_scores,
            risk_level=risk_level,
            recommendation=recommendation,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict grades for multiple students"""
    try:
        predictions = []
        grade_counts = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0}
        
        for i, student in enumerate(request.students):
            # Create prediction for each student
            df = pd.DataFrame([student.model_dump()])
            df = create_features(df)
            
            # Encode categorical
            for col in preprocessor['categorical_cols']:
                if col in df.columns:
                    df[col] = preprocessor['label_encoders'][col].transform(df[col].astype(str))
            
            # Prepare features
            feature_cols = preprocessor['categorical_cols'] + preprocessor['numerical_cols'] + \
                           ['average_score', 'total_score', 'study_efficiency', 'attendance_score']
            
            X = preprocessor['scaler'].transform(df[feature_cols])
            
            # Predict
            pred_class = model.predict(X)[0]
            pred_proba = model.predict_proba(X)[0]
            
            grade_names = preprocessor['label_encoders']['final_grade'].classes_
            predicted_grade = grade_names[pred_class]
            risk_level, recommendation = get_risk_level(predicted_grade)
            
            confidence_scores = {grade_names[i].upper(): float(pred_proba[i]) for i in range(len(grade_names))}
            
            predictions.append(PredictionResponse(
                student_id=i+1,
                predicted_grade=predicted_grade.upper(),
                predicted_grade_numeric=int(pred_class),
                confidence_scores=confidence_scores,
                risk_level=risk_level,
                recommendation=recommendation,
                prediction_timestamp=datetime.now().isoformat()
            ))
            
            grade_counts[predicted_grade] += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            summary=grade_counts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
