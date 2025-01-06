import librosa
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        harmony, perceptr = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        features = {
            'filename': file_path,
            'length': librosa.get_duration(y=y, sr=sr),
            'chroma_stft_mean': np.mean(chroma_stft),
            'chroma_stft_var': np.var(chroma_stft),
            'rms_mean': np.mean(rms),
            'rms_var': np.var(rms),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_var': np.var(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_var': np.var(spectral_bandwidth),
            'rolloff_mean': np.mean(rolloff),
            'rolloff_var': np.var(rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_var': np.var(zero_crossing_rate),
            'harmony_mean': np.mean(harmony),
            'harmony_var': np.var(harmony),
            'perceptr_mean': np.mean(perceptr),
            'perceptr_var': np.var(perceptr),
            'tempo': tempo,
        }

        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
            features[f'mfcc{i}_var'] = np.var(mfcc[i-1])

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

    return features

def process_batch(file_paths, genres):
    features_list = []
    labels = []
    for file_path, genre in zip(file_paths, genres):
        features = extract_features(file_path)
        if features is not None:
            features_list.append(features)
            labels.append(genre)
    return features_list, labels

def extract_features_in_batches(directory, batch_size=10, output_file="../data/gtzan_features_1.csv"):
    files = []
    genres = []

    for genre in os.listdir(directory):
        genre_dir = os.path.join(directory, genre)
        if os.path.isdir(genre_dir):
            for i, file_name in enumerate(os.listdir(genre_dir)):
                file_path = os.path.join(genre_dir, file_name)
                if file_path.endswith('.wav'):
                    files.append(file_path)
                    genres.append(genre)

    batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
    genre_batches = [genres[i:i+batch_size] for i in range(0, len(genres), batch_size)]

    all_features = []
    for i, (batch_files, batch_genres) in enumerate(zip(batches, genre_batches)):
        print(f"Processing batch {i+1}/{len(batches)}...")
        features, labels = process_batch(batch_files, batch_genres)

        batch_df = pd.DataFrame(features)
        batch_df['genre'] = labels
        all_features.append(batch_df)

        if os.path.exists(output_file):
            batch_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(output_file, index=False)

    final_df = pd.concat(all_features, ignore_index=True)
    return final_df
