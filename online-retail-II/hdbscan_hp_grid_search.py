import pandas as pd
import numpy as np
import sklearn as skl
import seaborn as sns
from tqdm import tqdm
from scipy.stats import randint, uniform, gamma
import random
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from collections import Counter
from hdbscan.validity import validity_index
import warnings
import dbcv
import umap

warnings.filterwarnings('ignore')



def dbcv_scorer(X, y):
    # dbcv_score = 0.0
    if np.unique(y) > 1:
        dbcv_score = dbcv.dbcv(X, y, metric='cosine')
    else:
        dbcv_score = -1
    return dbcv_score
# Function to run HDBSCAN with given hyperparameters and compute the scores
def run_hdbscan(params, X, dist_matrix):
    # min_cluster_size = params['min_cluster_size']
    # min_samples = params['min_samples']
    
    # Fit HDBSCAN
    # clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    # print(skl.__version__)
    clusterer = HDBSCAN(**params, n_jobs=6, metric='cosine')
    cluster_labels = clusterer.fit_predict(X)

    n_clusters = len(np.unique(cluster_labels)) - 1
    all_noise = np.array([-1])
    if not np.all(all_noise == np.unique(cluster_labels)):
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

        _params = {
            'min_cluster_size': params['min_cluster_size'],
            'min_samples': params['min_samples'],
            'num_noise_points': num_noise_points,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'dbcv_score': dbcv_score,
            'labels_': cluster_labels,
            'n_clusters': n_clusters
        }

        return _params
    else:
        return None


def randomized_search_hdbscan(
        estimator,
        X,
        distance_matrix,
        param_dict,
        n_iter,
        n_update):
    best_score = -np.inf
    best_params = None
    best_model = None
    max_noise = np.inf
    update_type = None
    for ind in tqdm(range(n_iter)):
        new_update = False
        # Randomly select parameters
        # params = {key: random.choice(values) for key, values in param_dict.items()}
        params = list(ParameterSampler(param_dict, n_iter=1))[0]
        # params['min_cluster_size'] = int(10**params['min_cluster_size']) + 1
        # Configure and fit the model
        # params['min_cluster_size'] = params['min_samples']
        if ind == 0:
            model = estimator(
                max_cluster_size=1000,
                n_jobs=6,
                metric='precomputed'
            )
        else:
            model = estimator(
                **params,
                max_cluster_size=1000,
                n_jobs=6,
                metric='precomputed'
            )

        labels = model.fit_predict(distance_matrix)
        n_noise = np.sum(labels < 0)
        # Calculate the DBCV score
        if np.sum(np.unique(labels) > -1) > 2:
            try:
                dbcv_res = validity_index(
                    distance_matrix,
                    labels,
                    metric='precomputed',
                    d=X.shape[1],
                    per_cluster_scores=False
                )
                # sd = 1.4826 * np.median(np.abs(dbcv_res[0] - dbcv_res[1]))
                score = dbcv_res# / sd
            except:
                score = -np.inf
            # try:
            #     score = dbcv.dbcv(X, labels, metric='cosine')
            # except:
            #     score = -1
        else:
            score = -np.inf

        # Update the best score and parameters
        if np.isfinite(score):
            if (score > best_score) & (n_noise < max_noise):
                # update_type = 'best'
                best_labels = labels
                best_score = score
                best_params = params
                best_model = model
                max_noise = n_noise
                new_update = True
            elif (score > 0.9 * best_score) & (n_noise < 1.1 * max_noise):
                best_labels = labels
                update_type = 'good'
                best_score = score
                best_params = params
                best_model = model
                max_noise = n_noise
                new_update = True
        # elif (score > best_score) & (max_noise - 0.25*max_noise < n_noise < max_noise + 0.05*max_noise):
        #     update_type = 'mid'
        #     best_score = score
        #     best_params = params
        #     best_model = model
        #     max_noise = n_noise

        if ((ind % n_update) == 0 or new_update) and np.isfinite(best_score):
            print(best_params)
            print(f'best DBCV score: {best_score:0.4f}')
            # print(f'n_clusters: {np.sum(np.unique(best_model.labels_) > -1)}')
            print(f'n_noise: {np.sum(best_model.labels_ < 0)}')
            print(f"noise types: {Counter(best_labels[best_labels < 0])}")
            # print(f'last update: {update_type}')
    return best_model, best_params, best_score


# Define the parameter grid
param_grid = {
    'min_cluster_size': np.arange(2, 20, 1).tolist(),
    'min_samples': np.arange(2, 20, 1).tolist(),
    'alpha': np.arange(0.5, 1.0, 0.001).tolist(),
    'leaf_size': np.arange(20, 60, 1).tolist(),
    'cluster_selection_method': ['leaf', 'eom'],
    'cluster_selection_epsilon': np.arange(0.1, .6, 0.001).tolist()
}
# param_grid = {
#     'min_samples': randint(2, 50),
#     'min_cluster_size': randint(2, 30),
#     'alpha': uniform(0.6, 0.4),
#     'leaf_size': randint(20, 60),
#     'cluster_selection_method': ['leaf', 'eom'],
#     'cluster_selection_epsilon': uniform(0.25, 0.25)
# }
# param_grid = {
#     'min_samples': randint(2, 50),
#     'min_cluster_size': uniform(0.05, 0.1),
#     'leaf_size': randint(20, 60),
#     'xi': uniform(0.01, 0.24),
#     # 'cluster_selection_epsilon': uniform(0.0, 1.0)
# }
embeddings = "embeddings.npy"
cosine_matrix = "embeddings_cosine_distance_matrix.npy"

with open(embeddings, 'rb') as f, open(cosine_matrix, 'rb') as g:
    X = np.load(f).astype('float64')
    distance_matrix = np.load(g).astype('float64')
    print(f"loaded {embeddings}, and {cosine_matrix}")

print(X.shape, distance_matrix.shape)
# assert X.shape[0] == X.shape[1]

# umaped_X = umap.UMAP(
#     n_neighbors=100,
#     min_dist=0.0,
#     n_components=189,
#     random_state=42,
# ).fit_transform(X)
#
#
# umap_cossim = cosine_similarity(umaped_X)
# scaled_X = skl.preprocessing.MinMaxScaler().fit_transform(umaped_X)
# distance_matrix = skl.preprocessing.MinMaxScaler().fit_transform(distance_matrix)
# Execute HDBSCAN runs in parallel
print("running HDBSCAN hyperparameter grid search")
best_mdl, best_params, best_score = randomized_search_hdbscan(
    HDBSCAN,
    X,
    distance_matrix,
    param_grid,
    n_iter=20000,
    n_update=100
)
print(best_params.items())
print(f'best DBCV score: {best_score:0.4f}')
print(f'n_clusters: {len(np.unique(best_mdl.labels_))-1}')
# results = []
# params_sets = []
# best_params = None
# best_dbcv = -np.inf
# for ii in tqdm(range(5000)):
#
#     params = list(skl.model_selection.ParameterSampler(param_grid, n_iter=1))[0]
#     res = run_hdbscan(params, X, distance_matrix)
#     if res is not None:
#         results.append(
#             {
#                 'labels_': res['labels_'],
#             }
#         )
#
#
#         if best_params is None:
#             best_params = params
#         if res['dbcv_score'] > best_dbcv:
#             best_params = params
#             best_dbcv = res['dbcv_score']
#
#         params_sets.append(params)
#         # params_sets[-1]['silhouette_score_'] = res['silhouette_score']
#         # params_sets[-1]['dbcv_score'] = res['dbcv_score']
#         # params_sets[-1]['n_clusters'] = res['n_clusters']
#
#
# print(best_dbcv)
# print(best_params)
# clusterer = HDBSCAN(n_jobs=1, metric='cosine')
# clusterer.fit(distance_matrix)

# fig, ax = plt.subplots(figsize=(10, 5))
# clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#
# ax.set_title("HDBSCAN DENDROGRAM")
# plt.savefig("hdbscan_dendrogram.png", bbox_inches='tight', dpi=150)
# plt.close()

# Convert results to DataFrame

# my_scorer = make_scorer(dbcv_scorer, greater_is_better=True)
# clf = RandomizedSearchCV(
#     estimator=clusterer,
#     param_distributions=param_grid,
#     n_iter=10,
#     scoring=my_scorer,
#     verbose=3,
#     cv=2
# )
# clf.fit(X)
# print(clf.best_params_)
# print(clf.best_score_)
# results_df = pd.DataFrame(results)
# params_df = pd.DataFrame(params_sets)
# Save results to CSV
# results_df.to_csv('hdbscan_results.csv', index=False)
# params_df.to_csv("hdbscan_params.csv", index=False)

# print(f"there were {params_df.shape[0]} good param sets.")
# print(f"best params with DBCV score of {params_df['dbcv_score'].max():0.3f}")
# print(params_df.iloc[params_df.dbcv_score.argmax()].T)
