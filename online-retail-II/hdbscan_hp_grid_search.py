import pandas as pd
import numpy as np
import sklearn as skl
from tqdm import tqdm
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# Function to run HDBSCAN with given hyperparameters and compute the scores
def run_hdbscan(params, X):
    min_cluster_size = params['min_cluster_size']
    min_samples = params['min_samples']
    
    # Fit HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=6)
    cluster_labels = clusterer.fit_predict(X)
    
    # Filter out noise points
    filtered_labels = cluster_labels[cluster_labels != -1]
    filtered_embeddings = X[cluster_labels != -1]
    
    if len(set(filtered_labels)) > 1:  # Check if there is more than one cluster
        silhouette_avg = silhouette_score(filtered_embeddings, filtered_labels)
        calinski_harabasz = calinski_harabasz_score(filtered_embeddings, filtered_labels)
        davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
    else:
        silhouette_avg = -1
        calinski_harabasz = -1
        davies_bouldin = float('inf')
    
    num_noise_points = np.sum(cluster_labels == -1)
    
    return {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'num_noise_points': num_noise_points,
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': calinski_harabasz,
        'davies_bouldin_score': davies_bouldin
    }

# Define the parameter grid
param_grid = [
    {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
    for min_cluster_size in range(5, 50, 5)
    for min_samples in range(1, 20, 2)
]

with open("embeddings_cosine_distance_matrix.npy", 'rb') as f:
    X = np.load(f)
    print("loaded embeddings cosine matrix")

assert X.shape[0] == X.shape[1]

# Execute HDBSCAN runs in parallel
print("running HDBSCAN hyperparameter grid search")
results = []
for ii, param_set in enumerate(tqdm(param_grid)):
    results.append(run_hdbscan(param_set, X))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('hdbscan_results.csv', index=False)

# Determine the best model based on defined criteria
best_model = results_df.loc[
    (results_df['num_noise_points'] == results_df['num_noise_points'].min()) &
    (results_df['silhouette_score'] == results_df['silhouette_score'].max()) &
    (results_df['calinski_harabasz_score'] == results_df['calinski_harabasz_score'].max()) &
    (results_df['davies_bouldin_score'] == results_df['davies_bouldin_score'].min())
]

# Print the best model's parameters and scores
print("Best HDBSCAN Model:")
print(best_model)

