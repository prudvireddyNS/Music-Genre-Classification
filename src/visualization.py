import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.iloc[:, 2:-1]
    labels = df.iloc[:, -1]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X)
    return features_scaled, labels

def pca_transform(features_scaled, n_components=3):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    return features_pca

def plot_3d_scatter(features_pca, labels, output_path):
    print("Starting 3D scatter plot...")
    unique_labels = pd.unique(labels)
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    colors = pd.Series([label_to_color[label] for label in labels])

    df = pd.DataFrame({
        'PC1': features_pca[:, 0],
        'PC2': features_pca[:, 1],
        'PC3': features_pca[:, 2],
        'Label': labels
    })

    fig = px.scatter_3d(
        df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Label',
        title="Interactive 3D Scatter Plot",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    print(f"Saving 3D scatter plot to {output_path}...")
    fig.write_html(output_path)
    print("3D scatter plot saved.")

def plot_pair_plot(df, output_path):
    pair_plot = sns.pairplot(df, hue='genre', palette='Set1')
    pair_plot.savefig(output_path)
    plt.close('all')

def plot_heatmap(df, output_path):
    corr = df.iloc[:, 2:-1].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(output_path)
    plt.close()

def plot_boxplot(df, output_path):
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df.iloc[:, 2:-1])
    plt.xticks(rotation=90)
    plt.title('Boxplot of Features')
    plt.savefig(output_path)
    plt.close()

def plot_violinplot(df, output_path):
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df.iloc[:, 2:-1])
    plt.xticks(rotation=90)
    plt.title('Violin Plot of Features')
    plt.savefig(output_path)
    plt.close()

def plot_countplot(df, output_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='genre', data=df)
    plt.title('Count Plot of Genres')
    plt.savefig(output_path)
    plt.close()

def main():
    df = load_data('./data/gtzan_features_1.csv')
    features_scaled, labels = preprocess_data(df)
    features_pca = pca_transform(features_scaled)

    # 3D Scatter Plot
    plot_3d_scatter(features_pca, labels, './plots/3d_scatter_plot.html')

    # Pair Plot
    # plot_pair_plot(df, './plots/pair_plot.png')

    # Heatmap
    plot_heatmap(df, './plots/heatmap.png')

    # Boxplot
    plot_boxplot(df, './plots/boxplot.png')

    # Violin Plot
    plot_violinplot(df, './plots/violinplot.png')

    # Count Plot
    plot_countplot(df, './plots/countplot.png')

if __name__ == "__main__":
    main()
