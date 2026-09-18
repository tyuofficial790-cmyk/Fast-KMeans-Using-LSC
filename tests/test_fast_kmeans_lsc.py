import numpy as np

from fast_kmeans_lsc import FastKMeansLSC


def test_kmeans_returns_expected_cluster_count():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 4))
    X[:100] += 3
    X[100:] -= 2

    model = FastKMeansLSC(n_clusters=2, n_landmarks=25, random_state=7)
    labels = model.fit_predict(X)

    assert len(labels) == len(X)
    assert set(np.unique(labels)) == {0, 1}
    assert model.cluster_centers_.shape == (2, 4)


def test_predict_works_after_fit():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ])
    model = FastKMeansLSC(n_clusters=2, n_landmarks=2, random_state=0)
    model.fit(X)

    prediction = model.predict(np.array([[0.05, 0.05], [10.05, 10.05]]))
    assert prediction.shape == (2,)
    assert set(prediction.tolist()) == {0, 1}
