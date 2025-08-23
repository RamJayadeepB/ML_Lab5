# lab05.py
# --------------------------------------------
# 23CSE301 - Lab 05 (Regression & Clustering)
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# ------------------- A1: Linear Regression (Single Feature) -------------------
def linear_regression_one_feature(df, feature_col, target_col):
    """
    Performs linear regression using a single feature.
    Returns: trained model, training data & predictions, testing data & predictions.
    """
    X = df[[feature_col]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return model, (X_train, y_train, y_train_pred), (X_test, y_test, y_test_pred)

# ------------------- A2: Evaluate Regression Model -------------------
def evaluate_regression(y_true, y_pred):
    """Calculates regression metrics: MSE, RMSE, MAPE, R2."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ------------------- A3: Linear Regression (Multiple Features) -------------------
def linear_regression_multiple_features(df, target_col, drop_cols=None):
    """
    Performs multiple linear regression using all features except target_col and drop_cols.
    """
    if drop_cols is None:
        drop_cols = []

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return model, (X_train, y_train, y_train_pred), (X_test, y_test, y_test_pred)

# ------------------- A4: KMeans Clustering -------------------
def kmeans_clustering(df, n_clusters):
    """
    Performs KMeans clustering on the dataset (target column should be removed beforehand).
    Returns: model, cluster labels, cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(df)
    return kmeans, kmeans.labels_, kmeans.cluster_centers_

# ------------------- A5: Evaluate Clustering -------------------
def clustering_scores(df, labels):
    """Computes Silhouette, Calinski-Harabasz, Davies-Bouldin scores for clustering."""
    sil = silhouette_score(df, labels)
    ch = calinski_harabasz_score(df, labels)
    db = davies_bouldin_score(df, labels)
    return {"Silhouette": sil, "Calinski-Harabasz": ch, "Davies-Bouldin": db}

# ------------------- A6: Clustering for Multiple k -------------------
def clustering_for_multiple_k(df, k_values):
    """Performs clustering for multiple k values and returns scores."""
    results = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(df)
        scores = clustering_scores(df, kmeans.labels_)
        results.append({"k": k, **scores})
    return results

# ------------------- A7: Elbow Method -------------------
def elbow_plot(df, k_range):
    """Plots the elbow curve (Distortion/Inertia vs. Number of clusters)."""
    distortions = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(df)
        distortions.append(km.inertia_)
    plt.plot(k_range, distortions, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.show()

# ------------------- MAIN PROGRAM -------------------
if __name__ == "__main__":
    # Load dataset 
    df = pd.read_csv("C:/Users/bramj/OneDrive/Desktop/Crop_recommendation.csv")

    # ========= A1 & A2: One Feature Regression =========
    model_one, train_data_one, test_data_one = linear_regression_one_feature(df, 'N', 'rainfall')
    _, y_train, y_train_pred = train_data_one
    _, y_test, y_test_pred = test_data_one

    metrics_train_one = evaluate_regression(y_train, y_train_pred)
    metrics_test_one = evaluate_regression(y_test, y_test_pred)

    print("\n=== A1 & A2: Linear Regression (1 Feature: N) on Rainfall ===")
    print("Train Metrics:", metrics_train_one)
    print("Test Metrics:", metrics_test_one)

    # ========= A3: Multiple Features Regression =========
    model_multi, train_data_multi, test_data_multi = linear_regression_multiple_features(df, 'rainfall', drop_cols=['label'])
    _, y_train_m, y_train_pred_m = train_data_multi
    _, y_test_m, y_test_pred_m = test_data_multi

    metrics_train_multi = evaluate_regression(y_train_m, y_train_pred_m)
    metrics_test_multi = evaluate_regression(y_test_m, y_test_pred_m)

    print("\n=== A3: Linear Regression (Multiple Features) on Rainfall ===")
    print("Train Metrics:", metrics_train_multi)
    print("Test Metrics:", metrics_test_multi)

    # ========= A4 & A5: KMeans Clustering (k=2) =========
    X_cluster = df.drop(columns=['label'])
    kmeans_model, labels, centers = kmeans_clustering(X_cluster, 2)
    cluster_scores_k2 = clustering_scores(X_cluster, labels)

    print("\n=== A4 & A5: KMeans Clustering (k=2) ===")
    print("Cluster Centers:\n", centers)
    print("Scores:", cluster_scores_k2)

    # ========= A6: Clustering Scores for Multiple k =========
    print("\n=== A6: Clustering Scores for multiple k ===")
    for res in clustering_for_multiple_k(X_cluster, range(2, 10)):
        print(res)

    # ========= A7: Elbow Plot =========
    print("\n=== A7: Elbow Plot ===")
    elbow_plot(X_cluster, range(2, 20))
