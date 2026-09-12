import numpy as np
from sklearn.decomposition import PCA
from eeg_lr.compare_pca import project


def test_explicit_pca_projection_matches_sklearn():
    rng=np.random.default_rng(7)
    train=rng.normal(size=(80,12))
    test=rng.normal(size=(10,12))
    model=PCA(n_components=8,svd_solver='full').fit(train)
    np.testing.assert_allclose(project(model,test,8),model.transform(test),atol=1e-12)
    np.testing.assert_allclose(project(model,test,3),model.transform(test)[:,:3],atol=1e-12)
    mean=model.mean_.copy()
    project(model,test+100,3)
    np.testing.assert_array_equal(model.mean_,mean)
