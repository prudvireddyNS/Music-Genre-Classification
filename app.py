import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd
from src.feature_extraction import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the trained model
model = joblib.load('./models/best_model.pkl')

# Load the scaler and PCA transformer
scaler = joblib.load('./models/scaler.pkl')
pca = joblib.load('./models/pca.pkl')

def preprocess_features(features):
    df = pd.DataFrame([features])
    features_scaled = scaler.transform(df.iloc[:, 2:])
    return features_scaled

def main():
    st.title("Music Genre Classification")
    st.write("Upload an audio file to predict its genre.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract features from the uploaded file
        features = extract_features("temp_audio.wav")

        if features is not None:
            # Preprocess the features
            features_pca = preprocess_features(features)

            # Make a prediction
            prediction = model.predict(features_pca)
            st.write(f"Predicted Genre: {prediction[0]}")
        else:
            st.write("Error processing the audio file.")

if __name__ == "__main__":
    main()
