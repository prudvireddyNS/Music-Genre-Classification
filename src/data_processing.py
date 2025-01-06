import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.iloc[:, 2:-1]
    labels = df.iloc[:, -1]

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X)
    return features_scaled, labels_encoded, label_encoder, scaler

def pca_transform(features_scaled, n_components=3):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    return features_pca

def split_data(features_scaled, labels, test_size=0.05, random_state=42):
    return train_test_split(features_scaled, labels, test_size=test_size, random_state=random_state)
