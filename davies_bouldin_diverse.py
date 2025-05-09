import numpy as np

from sklearn.utils import check_X_y, _safe_indexing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances

def check_number_of_labels(n_labels, n_samples):
    """
    Internal function sourced from scikit-learn: 
    https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/metrics/cluster/_unsupervised.py#L446
    
    Check that number of labels are valid.

    Parameters
    ----------
    n_labels : int
        Number of labels.

    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )
def db_diverse_score(X, labels, alpha=10.0):
    """
    Inspired by davies_bouldin_score from scikit-learn:
    https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/metrics/cluster/_unsupervised.py#L446

    Clustering score that rewards both:
    - Larger inter-cluster separation (centroid distance)
    - Larger intra-cluster diversity (spread)
        - This is the key difference from the standard Davies-Bouldin index

    Applies normalization and log scaling to ensure bounded, interpretable scores.
    Higher scores are better.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points.

    labels : array-like of shape (n_samples,)
        Cluster labels for each data point.

    alpha : float, optional (default=10.0)
        Scaling factor for log transformation.

    Returns
    -------
    score : float
        The resulting augmentation-friendly clustering score.
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, X.shape[1]), dtype=float)
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))

    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    combined_intra_dists = intra_dists[:, None] + intra_dists

    # Mask out self-comparisons
    mask = ~np.eye(n_labels, dtype=bool)
    valid_centroid_distances = centroid_distances[mask]
    valid_intra_dists = combined_intra_dists[mask]

    # Apply log scaling
    raw_scores = valid_centroid_distances * valid_intra_dists
    log_scores = np.log1p(alpha * raw_scores)

    return np.mean(log_scores)
