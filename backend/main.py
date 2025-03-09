from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
from pydantic import BaseModel
import shutil
# from feature_extraction import extract_features
from src.feature_extraction import extract_features

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models and transformers
model = joblib.load('./models/best_model.pkl')
scaler = joblib.load('./models/scaler.pkl')
pca = joblib.load('./models/pca.pkl')

label_map = {0: "blues", 
             1: "classical",
             2: "country",
             3: "disco",
             4: "hiphop",
             5: "jazz",
             6: "metal",
             7: "pop",
             8: "reggae",
             9: "rock"
             }

def preprocess_features(features):
    df = pd.DataFrame([features])
    features_scaled = scaler.transform(df.iloc[:, 2:])
    return features_scaled

@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        # Copy the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
    
    try:
        # Extract features
        features = extract_features(temp_path)
        
        if features is None:
            raise HTTPException(status_code=400, detail="Error processing audio file")
        
        # Preprocess features
        features_processed = preprocess_features(features)
        
        # Make prediction
        prediction = model.predict(features_processed)
        probabilities = model.predict(features_processed)
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        return {
            "genre": label_map[int(prediction[0])],
            # "confidence": float(confidence)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)