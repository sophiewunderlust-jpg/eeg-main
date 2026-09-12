"""Expanded subject-separated experiment; S001–080 develop, S081–109 test."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from threadpoolctl import threadpool_limits

from .compare_lda import lda
from .compare_pca import project
from .evaluate import bandpower_logreg, score_predictions, write_results
from .features import BandPowerFeatures
from .predict import PARAMETERS
from .prepare import prepare_recording

NAMES = ('logreg', 'lda', 'pca_8', 'pca_16', 'pca_32', 'pca_64')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def extract(raw_dir, subjects, audit):
    features=[]; labels=[]; groups=[]; orders=[]
    for subject in subjects:
        kept=0
        for run in (4,8,12):
            name=f'S{subject:03d}R{run:02d}.edf'
            path=raw_dir/f'S{subject:03d}'/name
            try:
                x,y,meta=prepare_recording(path,run,**PARAMETERS)
            except (ValueError, FileNotFoundError) as exc:
                # Fixed metadata/format policy, never a performance-based exclusion.
                audit[name]={'recording_error':str(exc).replace(str(raw_dir),'<data-dir>')}
                print(f'{name}: excluded ({type(exc).__name__})',flush=True)
                continue
            audit[name]={'events':meta['events']}
            if not len(x):
                continue
            f=BandPowerFeatures().fit_transform(x[:,:-1,:]*1e6)
            features.append(f); labels.append(y)
            groups.append(np.full(len(y),subject))
            orders.extend([f'{run}:{round((onset-4.2)/8.3)}' for onset in meta['kept_onsets']])
            kept+=len(y)
        print(f'S{subject:03d}: {kept} accepted trials',flush=True)
    if not features:
        raise ValueError('No usable recordings')
    return np.concatenate(features),np.concatenate(labels),np.concatenate(groups),np.array(orders).reshape(-1,1)


def fit_classifier(x,y,name):
    scaler=StandardScaler().fit(x)
    transformed=np.clip(scaler.transform(x),-20,20)
    pca=None
    if name.startswith('pca_'):
        k=int(name.split('_')[1])
        pca=PCA(n_components=k,svd_solver='full',whiten=False).fit(transformed)
        transformed=project(pca,transformed,k)
    model=bandpower_logreg().steps[-1][1] if name=='logreg' else lda()
    model.fit(transformed,y)
    return scaler,pca,model


def predict(fitted,x):
    scaler,pca,model=fitted
    transformed=np.clip(scaler.transform(x),-20,20)
    if pca is not None:
        transformed=project(pca,transformed,len(pca.components_))
    if not np.isfinite(model.decision_function(transformed)).all():
        raise ValueError('Non-finite classifier scores')
    return model.predict(transformed)


def scored(name,split,y_train,y,pred,subjects):
    rows=[]
    excluded=[]
    for subject in np.unique(subjects):
        mask=subjects==subject
        if len(np.unique(y[mask]))!=2:
            excluded.append(int(subject))
            continue
        rows.append(score_predictions(model=name,split=split,held_out=f'S{subject:03d}',
                    y_train=y_train,y_test=y[mask],predictions=pred[mask]))
    return rows,excluded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('reports/expanded'))
    args=parser.parse_args()
    out=args.output
    if (out/'selection.json').exists() or (out/'test/summary.json').exists():
        parser.error('Output already contains an experiment; choose a new output directory to avoid overwriting results.')
    protocol={'development_subjects':list(range(1,81)),'test_subjects':list(range(81,110)),
        'preprocessing':PARAMETERS,'runs':[4,8,12],'features':'126 mu/beta log-band-power values; no CSP in this expansion',
        'candidates':list(NAMES),'validation':'5-fold GroupKFold; person-separated; equal-person mean balanced accuracy',
        'selection':'Maximum development mean; exact ties favor earlier candidate in listed order',
        'test_comparators':['selected trained on S001–080','same configuration trained on S001–010','majority','run/cue ordinal only'],
        'exclusions':'Unsupported metadata/files excluded with recording audit; subjects lacking both classes excluded from balanced-accuracy means',
        'note':'S011–030 are now development data. S081–109 are untouched until selection is saved. No test-based tuning.'}
    save(out/'protocol.json',protocol)
    warnings.filterwarnings('error',category=RuntimeWarning)
    with threadpool_limits(limits=1):
        audit={}
        x,y,s,order=extract(args.data_dir,range(1,81),audit)
        save(out/'development_event_audit.json',audit)
        results=[]; fold_manifest=[]
        for fold,(tr,va) in enumerate(GroupKFold(n_splits=5).split(x,y,s),1):
            print(f'Validation fold {fold}/5',flush=True)
            assert not set(s[tr]) & set(s[va])
            fold_manifest.append({'fold':fold,'training_subjects':np.unique(s[tr]).tolist(),
                                  'validation_subjects':np.unique(s[va]).tolist()})
            for name in NAMES:
                fit=fit_classifier(x[tr],y[tr],name)
                rows,_=scored(name,'expanded-development',y[tr],y[va],predict(fit,x[va]),s[va])
                results.extend(rows)
        write_results(results,out/'development')
        save(out/'folds.json',fold_manifest)
        means={name:float(np.mean([v.balanced_accuracy for v in results if v.model==name])) for name in NAMES}
        selected=max(NAMES,key=lambda name:means[name])
        selection={'selected':selected,'development_mean_balanced_accuracy':means,'train_epochs':len(y),
                   'actual_training_subjects':np.unique(s).tolist()}
        save(out/'selection.json',selection)
        print(f'SELECTION FROZEN: {selected}: {means[selected]*100:.2f}%',flush=True)
        full=fit_classifier(x,y,selected)
        small=s<=10
        fit10=fit_classifier(x[small],y[small],selected)
        # Test files are first read after development-only selection is committed to disk.
        audit_test={}
        xt,yt,st,ot=extract(args.data_dir,range(81,110),audit_test)
        save(out/'test_event_audit.json',audit_test)
        predictions={'selected_80':predict(full,xt),'same_model_10':predict(fit10,xt),
                     'majority':np.full(len(yt),int(np.bincount(y).argmax()))}
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(order)
        nuisance=bandpower_logreg().steps[-1][1].fit(encoder.transform(order),y)
        predictions['timing_only']=nuisance.predict(encoder.transform(ot))
        test_rows=[]; excluded=[]
        for name,pred in predictions.items():
            rows,excluded=scored(name,'expanded-test',y[small] if name=='same_model_10' else y,yt,pred,st)
            test_rows.extend(rows)
        write_results(test_rows,out/'test')
        # Preserve per-trial predictions for all comparators for a reproducible audit.
        save(out/'test_predictions.json',{'subjects':st.tolist(),'truth':yt.tolist(),
                                         'predictions':{k:v.tolist() for k,v in predictions.items()}})
    rng=np.random.default_rng(7)
    values={name:np.array([r.balanced_accuracy for r in test_rows if r.model==name]) for name in predictions}
    boot=rng.integers(0,len(values['selected_80']),size=(10000,len(values['selected_80'])))
    ci=np.quantile(values['selected_80'][boot].mean(axis=1),[.025,.975])
    diff=values['selected_80']-values['same_model_10']
    diff_ci=np.quantile(diff[boot].mean(axis=1),[.025,.975])
    stats={'selected':selected,'train_epochs':len(y),'test_epochs':len(yt),
        'scored_test_subjects':len(values['selected_80']),'single_class_subjects_excluded_from_scoring':excluded,
        'missing_test_subjects':sorted(set(range(81,110))-set(st.tolist())),
        'selected_test_mean':float(values['selected_80'].mean()),'selected_subject_bootstrap_95_ci':ci.tolist(),
        'paired_gain_80_vs_10':float(diff.mean()),'paired_gain_bootstrap_95_ci':diff_ci.tolist(),
        'bootstrap_seed':7,'bootstrap_replicates':10000}
    save(out/'experiment.json',stats)
    report=['# Expanded experiment: up to 80 training participants','',
        'S001–S080 are development/training; S081–S109 were kept untouched until development selection. '
        'Subjects previously used for evaluation, S011–S030, are explicitly reassigned to development. '
        'Preprocessing and the six candidates were fixed before this experiment. No test-driven model changes.', '',
        '## Development selection','',
        'Five-fold subject-separated validation. Each usable person has equal weight in the mean. '
        'These scores select a model; they are not unbiased estimates for the selected model.', '',
        '| Model | Balanced accuracy |','|---|---:|']
    report += [f'| {name} | {value*100:.2f}% |' for name,value in means.items()]
    report += ['',f'Frozen choice: **{selected}**. Training trials: **{len(y)}**; test trials: **{len(yt)}**. '
        f"Scored test people: **{stats['scored_test_subjects']}**. Metadata exclusions are preserved in the recording audits.",'',
        '## One final test on S081–S109','',
        '| Training / model | Balanced accuracy |','|---|---:|']
    report += [f'| {name} | {value.mean()*100:.2f}% |' for name,value in values.items()]
    report += ['',f'Selected-model 95% subject-bootstrap interval: **{ci[0]*100:.2f}–{ci[1]*100:.2f}%**.', '',
        f'Same-model comparison on the same test people: training on up to 80 versus 10 gives '
        f'**{diff.mean()*100:+.2f} percentage points**, paired 95% bootstrap interval '
        f'**[{diff_ci[0]*100:+.2f}, {diff_ci[1]*100:+.2f}]**.', '',
        'The ten-person comparator uses the configuration selected using 80 development people, '
        'so it isolates the final fitting-data difference conditional on that configuration, not the entire model-selection process.', '',
        '## Scope and limitations','',
        'Only band-power candidates are included: logistic regression, shrinkage LDA, and PCA (8/16/32/64) + shrinkage LDA. '
        'Preprocessing, standardization, clipping, PCA, and LDA settings match the previous comparisons. '
        'Only learned transforms fit inside training folds; deterministic filtering/feature extraction is per recording. '
        'CSP is not retrained in this focused expansion. '
        'Subject IDs form a non-random development/test split. Trial rejection and unsupported-recording exclusions '
        'limit the population represented. No clinical, online, or neural-specific performance guarantee is made.', '',
        'The original exported model is not changed. The prediction CLI still uses the original logistic-regression artifact. '
        'See protocol.json, selection.json, folds.json, the per-person score files and event audits.', '',
        '```bash','python -m eeg_lr.expand_experiment --data-dir /path/to/data','```']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report),flush=True)


if __name__=='__main__':
    main()
