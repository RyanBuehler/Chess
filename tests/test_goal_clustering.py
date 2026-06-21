import numpy as np
from chessrl.goals.clustering import kmeans_fit, assign_nearest, median_radius


def _three_blobs(rng):
    a = rng.normal([0, 0], 0.05, size=(100, 2))
    b = rng.normal([5, 5], 0.05, size=(100, 2))
    c = rng.normal([0, 5], 0.05, size=(100, 2))
    return np.vstack([a, b, c]).astype(np.float32)


def test_recovers_blob_centers():
    rng = np.random.default_rng(0)
    x = _three_blobs(rng)
    cents = kmeans_fit(x, k=3, rng=rng)
    assert cents.shape == (3, 2)
    # every true center has a near centroid
    for tc in ([0, 0], [5, 5], [0, 5]):
        d = np.linalg.norm(cents - np.array(tc), axis=1).min()
        assert d < 0.5, (tc, cents)


def test_assign_nearest_labels():
    cents = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    x = np.array([[0.1, 0.0], [9.9, 10.1]], dtype=np.float32)
    labels = assign_nearest(x, cents)
    assert labels.tolist() == [0, 1]


def test_always_returns_k_centroids_even_with_few_points():
    rng = np.random.default_rng(1)
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)  # n=2 < k=5
    cents = kmeans_fit(x, k=5, rng=rng)
    assert cents.shape == (5, 2)


def test_no_empty_clusters_after_fit():
    rng = np.random.default_rng(2)
    x = _three_blobs(rng)
    cents = kmeans_fit(x, k=5, rng=rng)  # more clusters than blobs
    labels = assign_nearest(x, cents)
    # every centroid index appears (reseeding prevents empties)
    assert set(labels.tolist()) == set(range(5))


def test_median_radius_nonneg():
    rng = np.random.default_rng(3)
    x = _three_blobs(rng)
    cents = kmeans_fit(x, k=3, rng=rng)
    labels = assign_nearest(x, cents)
    tau = median_radius(x, cents, labels)
    assert tau >= 0.0 and np.isfinite(tau)
