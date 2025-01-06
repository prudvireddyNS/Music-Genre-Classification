import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_processing import load_data, preprocess_data, split_data
from feature_extraction import extract_features_in_batches
from model_training import train_svm, train_knn, train_decision_tree, train_neural_network
from evaluation import evaluate_model

def main():
    directory = "./data/genres_original"
    batch_size = 10
    output_file = "./data/gtzan_features_1.csv"

    # Extract features and save to CSV
    # df = extract_features_in_batches(directory, batch_size, output_file)

    # Load data
    df = load_data(output_file)

    # Preprocess data
    features_scaled, labels, label_encoder, scaler = preprocess_data(df)
    joblib.dump(scaler, './models/scaler.pkl')
    joblib.dump(label_encoder, './models/label_encoder.pkl')

    # Save the scaler
    # scaler = StandardScaler()
    # scaler.fit(df.iloc[:, 2:-1])
    # joblib.dump(scaler, './models/scaler.pkl')

    # # Save the PCA transformer
    # pca = PCA(n_components=3)
    # pca.fit(features_scaled)
    # joblib.dump(pca, './models/pca.pkl')

    # Split data
    X_train, X_test, y_train, y_test = split_data(features_scaled, labels)

    # Train models
    svm_model = train_svm(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    nn_model = train_neural_network(X_train, y_train)

    # Evaluate models
    print("Evaluating SVM Model:")
    evaluate_model(svm_model, X_test, y_test)

    print("Evaluating KNN Model:")
    evaluate_model(knn_model, X_test, y_test)

    print("Evaluating Decision Tree Model:")
    evaluate_model(dt_model, X_test, y_test)

    print("Evaluating Neural Network Model:")
    evaluate_model(nn_model, X_test, y_test)

    # Save the best model
    best_model = svm_model  # Choose the best model based on evaluation
    joblib.dump(best_model, './models/best_model.pkl')

if __name__ == "__main__":
    main()
