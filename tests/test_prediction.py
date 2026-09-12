from pathlib import Path
import numpy as np
from eeg_lr.dataset import CHANNELS
from eeg_lr.evaluate import bandpower_logreg
from eeg_lr.predict import export_model, predict_arrays, predict_edf
import pytest


def test_export_matches_fitted_model(tmp_path):
    rng=np.random.default_rng(17)
    x=rng.normal(scale=10e-6,size=(20,64,561))
    y=np.tile([0,1],10)
    model=bandpower_logreg().fit(x[:,:-1]*1e6,y)
    path=tmp_path/'model.npz'
    export_model(model,path,list(CHANNELS))
    predicted, probability=predict_arrays(x,path)
    assert np.array_equal(predicted,model.predict(x[:,:-1]*1e6))
    assert np.isfinite(probability).all()
    assert ((probability>=0)&(probability<=1)).all()


def test_rejects_unrelated_run():
    with pytest.raises(ValueError,match='imagined'):
        predict_edf(Path('S001R06.edf'),Path('unused.npz'))


def test_prediction_does_not_use_cue_class(tmp_path, monkeypatch):
    import mne
    import eeg_lr.prepare as prepare
    rng=np.random.default_rng(5)
    training=rng.normal(scale=10e-6,size=(20,64,561))
    model=bandpower_logreg().fit(training[:,:-1]*1e6,np.tile([0,1],10))
    path=tmp_path/'model.npz'
    export_model(model,path,list(CHANNELS))
    raw=mne.io.RawArray(rng.normal(scale=10e-6,size=(64,1920)),
                       mne.create_info(list(CHANNELS),160,'eeg'),verbose=False)
    raw.set_annotations(mne.Annotations([1,7],[4.1,4.1],['T1','T2']))
    monkeypatch.setattr(prepare,'load_raw',lambda *a,**k: raw.copy())
    before=predict_edf(Path('S001R04.edf'),path)
    with_targets=predict_edf(Path('S001R04.edf'),path,show_targets=True)
    assert [r['correct_label'] for r in with_targets['predictions']]==['left_fist','right_fist']
    for plain, annotated in zip(before['predictions'],with_targets['predictions']):
        assert all(annotated[k]==v for k,v in plain.items())
        assert annotated['correct']==(annotated['predicted_label']==annotated['correct_label'])
    raw.set_annotations(mne.Annotations([1,7],[4.1,4.1],['T2','T1']))
    after=predict_edf(Path('S001R04.edf'),path)
    assert before==after
    assert before['accepted_trials']==2
