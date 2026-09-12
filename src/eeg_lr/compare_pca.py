"""Exploratory, development-selected PCA + shrinkage LDA on band power."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .compare_lda import lda
from .evaluate import bandpower_logreg, load_prepared, score_predictions, write_results

COUNTS = (8, 16, 32, 64)


def project(pca, x, count):
    """Unwhitened PCA projection, explicit product for this Mac runtime."""
    if x.ndim != 2 or x.shape[1] != len(pca.mean_) or not 1 <= count <= len(pca.components_):
        raise ValueError('PCA projection shape/component mismatch')
    return np.einsum('nf,kf->nk', x-pca.mean_, pca.components_[:count])


def fit_predict(train_x, train_y, test_x, counts=COUNTS):
    transform = Pipeline(bandpower_logreg().steps[:-1])
    train = transform.fit_transform(train_x, train_y)
    test = transform.transform(test_x)
    # Full SVD is deterministic. The first k vectors are identical whether the
    # full decomposition is retained or truncated to k; no labels enter PCA.
    pca = PCA(n_components=max(counts), svd_solver='full', whiten=False).fit(train)
    output = {}
    for k in (None, *counts):
        a, b = (train, test) if k is None else (project(pca, train, k), project(pca, test, k))
        model = lda().fit(a, train_y)
        scores = model.decision_function(b)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite PCA/LDA scores')
        output['no_pca' if k is None else f'pca_{k}'] = model.predict(b)
    variance = {str(k): float(pca.explained_variance_ratio_[:k].sum()) for k in counts}
    return output, variance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=Path('data/final-development'))
    parser.add_argument('--test', type=Path, default=Path('data/final-heldout'))
    parser.add_argument('--output', type=Path, default=Path('reports/pca-lda-comparison'))
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('error', category=RuntimeWarning)
    with threadpool_limits(limits=1):
        x,y,s,r=load_prepared(args.development)
        if set(s)!=set(range(1,11)):
            raise ValueError('Expected S001–010 development subjects')
        results=[]
        variances={}
        for tr,te in LeaveOneGroupOut().split(x,y,s):
            subject=f'S{s[te][0]:03d}'
            print(f'Development {subject}', flush=True)
            predictions,variance=fit_predict(x[tr],y[tr],x[te])
            variances[subject]=variance
            for name,pred in predictions.items():
                results.append(score_predictions(model=name, split='subject-grouped', held_out=subject,
                    y_train=y[tr],y_test=y[te],predictions=pred))
        write_results(results,args.output/'development')
        scores={name: float(np.mean([v.balanced_accuracy for v in results if v.model==name]))
                for name in ('no_pca',*[f'pca_{k}' for k in COUNTS])}
        selected=min(COUNTS,key=lambda k:(-scores[f'pca_{k}'],k))
        selection={'candidates':list(COUNTS),'selected_components':selected,
                   'rule':'Highest development mean subject-grouped balanced accuracy; ties favor fewer components.',
                   'development_scores':scores,'fold_explained_variance':variances,
                   'caveat':'Post hoc follow-up; S011–030 have been inspected in earlier experiments. No exported model replacement.'}
        # Persist development-only selection before loading follow-up test data.
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        print(f'Selected {selected} components using development only',flush=True)
        xt,yt,st,_=load_prepared(args.test)
        if set(st)!=set(range(11,31)):
            raise ValueError('Expected S011–030 follow-up subjects')
        predictions,variance=fit_predict(x,y,xt,(selected,))
        followup=[]
        for name,pred in predictions.items():
            for subject in np.unique(st):
                mask=st==subject
                followup.append(score_predictions(model=name,split='followup-heldout',held_out=f'S{subject:03d}',
                    y_train=y,y_test=yt[mask],predictions=pred[mask]))
        write_results(followup,args.output/'followup-heldout')
    base={v.held_out:v.balanced_accuracy for v in followup if v.model=='no_pca'}
    chosen={v.held_out:v.balanced_accuracy for v in followup if v.model==f'pca_{selected}'}
    diff=np.array([chosen[k]-base[k] for k in sorted(base)])
    rng=np.random.default_rng(7)
    ci=np.quantile(rng.choice(diff,size=(10000,len(diff)),replace=True).mean(axis=1),[.025,.975])
    paired={'mean_difference':float(diff.mean()),'paired_subject_bootstrap_95_ci':ci.tolist(),
            'wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'losses':int((diff<0).sum()),
            'seed':7,'replicates':10000,'full_train_explained_variance':variance}
    (args.output/'paired_differences.json').write_text(json.dumps(paired,indent=2)+'\n')
    lines=['# PCA before shrinkage LDA: exploratory follow-up','',
        'Pipeline: EDF preprocessing → 126 mu/beta log-band-power features → training-only standardization '
        'and fixed ±20 clipping → PCA (full SVD, no whitening) → automatic-shrinkage LDA. '
        'The no-PCA baseline uses identical preprocessing and LDA. PCA centers its input and retains '
        'directions with the most variance, not necessarily the most left/right information.', '',
        'PCA and scaling are fitted separately inside each development training fold. '
        'Fixed candidate counts are 8, 16, 32 and 64. The best count is selected using S001–010 '
        'leave-one-person-out balanced accuracy; ties favor fewer components. '
        'Selected development performance is tuning performance, not an unbiased generalization estimate.', '',
        '| Representation | Development mean balanced accuracy |','|---|---:|']
    for name,score in scores.items():
        lines.append(f'| {name} | {score*100:.2f}% |')
    lines += ['',f'Selected PCA setting: **{selected} components**, retaining '
              f'**{variance[str(selected)]*100:.1f}%** of variance when fitted on all development data.', '',
              '| Model | Follow-up S011–030 balanced accuracy |','|---|---:|',
              f'| No PCA + LDA | {np.mean(list(base.values()))*100:.2f}% |',
              f'| PCA ({selected}) + LDA | {np.mean(list(chosen.values()))*100:.2f}% |','',
              f'Mean paired change: **{diff.mean()*100:+.2f} percentage points**; '
              f'95% paired subject-bootstrap interval **[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]**. '
              f"Better for {paired['wins']}, tied for {paired['ties']}, worse for {paired['losses']} people.", '',
              'This is exploratory: S011–030 results were already seen before requesting PCA. '
              'Only the development-selected PCA setting is evaluated here; test results do not select the component count. '
              'No new untouched subjects, multiplicity-adjusted significance, or production-model replacement is claimed. '
              '50% is chance balanced accuracy; a small improvement is not reliable neural decoding.', '',
              'The PCA projection uses the algebraically equivalent explicit NumPy einsum product '
              'to avoid the previously observed matrix-product warning. Its equivalence to sklearn PCA.transform is tested.', '',
              'Reproduce (after eeg-finalize creates prepared folders):', '',
              '```bash','python -m eeg_lr.compare_pca','```','',
              'Reference: https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.PCA.html']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
