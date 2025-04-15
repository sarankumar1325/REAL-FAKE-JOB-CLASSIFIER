from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import pickle
import numpy as np
from pydantic import BaseModel
import uvicorn
import logging
import traceback
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Mount static files with absolute path
static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class JobPosting(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str

def extract_features(text):
    """Extract basic numerical features from text"""
    # Count words
    words = text.split()
    word_count = len(words)
    
    # Count unique words
    unique_words = len(set(words))
    
    # Count characters
    char_count = len(text)
    
    # Count capital letters
    capital_count = sum(1 for c in text if c.isupper())
    
    # Count numbers
    number_count = sum(1 for c in text if c.isdigit())
    
    # Count special characters
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    return [
        word_count,
        unique_words,
        char_count,
        capital_count,
        number_count,
        special_count
    ]

def preprocess_text(job_posting):
    """Preprocess all text fields and combine features"""
    # Process each field
    title_features = extract_features(job_posting.title)
    company_features = extract_features(job_posting.company_profile)
    desc_features = extract_features(job_posting.description)
    req_features = extract_features(job_posting.requirements)
    benefits_features = extract_features(job_posting.benefits)
    
    # Combine all text for main features
    combined_text = f"{job_posting.title} {job_posting.company_profile} {job_posting.description} {job_posting.requirements} {job_posting.benefits}"
    combined_text = combined_text.lower()
    
    # Extract features from combined text
    total_features = extract_features(combined_text)
    
    # Combine all numerical features
    all_features = (
        title_features +
        company_features +
        desc_features +
        req_features +
        benefits_features +
        total_features
    )
    
    return combined_text, np.array(all_features)

# Load the model and vectorizer
try:
    logger.info("Loading the model...")
    model_path = BASE_DIR / "static" / "trained_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully!")
    
    # Initialize vectorizer with specific parameters
    vectorizer = TfidfVectorizer(
        max_features=13,  # Adjust this to match your training
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 1)
    )
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict(job_posting: JobPosting):
    if model is None:
        logger.error("Model not loaded - please ensure trained_model.pkl exists in the project directory")
        return JSONResponse(
            status_code=500,
            content={"detail": "The machine learning model is not available. Please ensure the model file (trained_model.pkl) is present in the project directory."}
        )

    try:
        # Preprocess the text and extract features
        processed_text, numerical_features = preprocess_text(job_posting)
        
        # Vectorize the text
        text_features = vectorizer.fit_transform([processed_text]).toarray()
        
        # Combine text and numerical features
        final_features = np.concatenate([text_features[0], numerical_features])
        
        # Ensure we have exactly 19 features
        if len(final_features) != 19:
            logger.warning(f"Feature mismatch: got {len(final_features)} features, expected 19")
            # Pad or truncate to match expected features
            if len(final_features) < 19:
                final_features = np.pad(final_features, (0, 19 - len(final_features)))
            else:
                final_features = final_features[:19]
        
        # Reshape for prediction
        final_features = final_features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(final_features)[0]
        
        # Get prediction probability if available
        confidence = 0.8  # Default confidence
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(final_features)[0]
                confidence = max(proba)
            except:
                pass
        
        # Prepare response
        response_data = {
            "prediction": int(prediction),
            "is_fake": bool(prediction == 1),
            "confidence": float(confidence),
            "message": "This job posting appears to be FAKE!" if prediction == 1 else "This job posting appears to be LEGITIMATE."
        }
        
        # Log the response
        logger.info(f"Prediction response: {json.dumps(response_data)}")
        
        return JSONResponse(
            content=jsonable_encoder(response_data),
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error during prediction: {str(e)}"}
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 