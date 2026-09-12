import numpy as np
from threadpoolctl import threadpool_limits
from eeg_lr.expand_experiment import NAMES, fit_classifier, predict, scored


def test_expanded_classifiers_and_scoring():
    rng=np.random.default_rng(32)
    x=rng.normal(size=(90,126))
    y=np.arange(90)%2
    with threadpool_limits(limits=1):
        for name in NAMES:
            fitted=fit_classifier(x[:80],y[:80],name)
            result=predict(fitted,x[80:])
            assert result.shape==(10,)
            assert np.isin(result,[0,1]).all()
            assert fitted[0].n_samples_seen_==80
    rows,excluded=scored('test','test',y,y[:6],y[:6],np.array([1,1,1,1,2,3]))
    assert len(rows)==1
    assert rows[0].held_out=='S001'
    assert rows[0].balanced_accuracy==1
    assert excluded==[2,3]
