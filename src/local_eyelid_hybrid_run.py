"""Run one private prespecified hybrid CV and reproduce saved decisions offline."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .local_eyelid_hybrid_features import PREPROCESSING, FrozenMobileNet, extract_features, private_output_path, validated_inputs
from .local_eyelid_hybrid_model import CANDIDATES, DESIGN, PARAMETERS, SEED, fit_outer_training, select_candidate, select_threshold
from .local_eyelid_hybrid_evaluation import METRICS, conditional_intervals, evaluate
from .local_eyelid_validation import checked_file, sha256, strict_csv
from .preserve_local_experiment import copy_verified, tree_checksums


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    path.chmod(0o600)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('x', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path.chmod(0o600)


def check_projection(model: Any, x: np.ndarray, candidate: str) -> None:
    projection=model.named_steps['projection']
    np.testing.assert_allclose(projection.scaler_.mean_,x[:,6:].mean(axis=0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(projection.scaler_.var_,x[:,6:].var(axis=0),rtol=0,atol=1e-12)
    if projection.training_samples_!=len(x) or projection.pca_.n_samples_!=len(x) or projection.pca_.components_.shape!=(8,576):
        raise ValueError('Projection training scope/dimensions differ')
    standardized=projection.scaler_.transform(x[:,6:])
    np.testing.assert_allclose(projection.pca_.mean_,standardized.mean(axis=0),rtol=0,atol=1e-12)
    scores=projection.pca_.transform(standardized)
    np.testing.assert_allclose(np.sum(scores*scores,axis=0)/(len(x)-1),projection.pca_.explained_variance_,rtol=1e-10,atol=1e-10)
    params=model.named_steps['classifier'].get_params()
    if any(params[key]!=value for key,value in PARAMETERS.items()) or params['class_weight']!=('balanced' if candidate=='balanced' else None):
        raise ValueError('Unexpected model settings')


def verify_models(output: Path) -> dict[str, Any]:
    """Reproduce all 35 saved pipelines, inner decisions and outer metrics."""
    configuration=json.loads((output/'experiment_config.json').read_text())
    if configuration['design']!=DESIGN or configuration['preprocessing']!=PREPROCESSING:
        raise ValueError('Run design differs from prespecified implementation')
    rows=strict_csv(output/'oof_predictions.csv')
    subjects=[r['subject_id'] for r in rows]
    y=np.array([int(r['label']) for r in rows]);folds=np.array([int(r['fold']) for r in rows])
    if len(subjects)!=len(set(subjects)) or subjects!=sorted(subjects) or len(subjects)!=119 or y.sum()!=19:
        raise ValueError('Original cohort coverage differs')
    with np.load(output/'features.npz',allow_pickle=False) as cache:
        x=cache['features'];cached_subjects=cache['subjects'].tolist()
    if cached_subjects!=subjects or x.shape!=(119,582) or not np.isfinite(x).all():
        raise ValueError('Cached feature coverage differs')
    lab_rows=strict_csv(output/'baseline_features.csv')
    from .local_eyelid_classifier import validate_feature_rows
    np.testing.assert_array_equal(x[:,:6],validate_feature_rows(lab_rows,subjects))
    copied_folds={r['subject_id']:int(r['fold']) for r in strict_csv(output/'subject_folds.csv')}
    if copied_folds!={s:int(f) for s,f in zip(subjects,folds)} or set(folds)!={1,2,3,4,5}:
        raise ValueError('Copied baseline fold assignment differs')
    p=np.array([float(r['hybrid_probability']) for r in rows])
    thresholds=np.array([float(r['selected_threshold']) for r in rows])
    baseline=np.array([float(r['baseline_probability']) for r in rows])
    for number in range(1,6):
        training=np.flatnonzero(folds!=number);validation=np.flatnonzero(folds==number)
        directory=output/f'fold_{number}'
        record=json.loads((directory/'selection.json').read_text())
        ids=[subjects[i] for i in training]
        if record['training_subjects']!=ids or record['training_labels']!=y[training].tolist():
            raise ValueError('Outer training membership differs')
        inner_y=y[training];inner_x=x[training]
        splits=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(inner_x,inner_y))
        inner_scores={}
        for candidate in CANDIDATES:
            probabilities=np.full(len(training),np.nan)
            for index,(fit,valid) in enumerate(splits,1):
                partition=record['inner_partitions'][index-1]
                if partition['training_subjects']!=[ids[i] for i in fit] or partition['validation_subjects']!=[ids[i] for i in valid] or set(fit)&set(valid):
                    raise ValueError('Inner subject partition differs')
                if partition['training_class_counts']!=np.bincount(inner_y[fit],minlength=2).tolist() or partition['validation_class_counts']!=np.bincount(inner_y[valid],minlength=2).tolist():
                    raise ValueError('Inner class support differs')
                model=joblib.load(directory/f'inner_{candidate}_{index}.joblib')
                check_projection(model,inner_x[fit],candidate)
                probabilities[valid]=model.predict_proba(inner_x[valid])[:,1]
                if not np.all(np.array(record['inner_assignment'])[valid]==index):
                    raise ValueError('Saved inner assignment differs')
            np.testing.assert_allclose(probabilities,record['inner_oof_probabilities'][candidate],rtol=0,atol=1e-14)
            inner_scores[candidate]=float(average_precision_score(inner_y,probabilities))
            np.testing.assert_allclose(inner_scores[candidate],record['inner_average_precision'][candidate],rtol=0,atol=1e-14)
        candidate=select_candidate(inner_scores)
        threshold=select_threshold(inner_y,np.array(record['inner_oof_probabilities'][candidate]))
        if candidate!=record['selected_candidate'] or any(record[key]!=value for key,value in threshold.items()) or not np.all(thresholds[validation]==record['threshold']):
            raise ValueError('Selected candidate/threshold differs')
        model=joblib.load(directory/'outer_model.joblib');check_projection(model,x[training],candidate)
        np.testing.assert_allclose(model.predict_proba(x[validation])[:,1],p[validation],rtol=0,atol=1e-14)
    if any(int(r['fixed_prediction'])!=int(float(r['hybrid_probability'])>=.5) or int(r['selected_prediction'])!=int(float(r['hybrid_probability'])>=float(r['selected_threshold'])) for r in rows):
        raise ValueError('Saved decision labels differ')
    metrics=json.loads((output/'metrics.json').read_text())
    for name,t in [('fixed_0_5',.5),('inner_selected',thresholds)]:
        if evaluate(y,p,t)!=metrics['pooled_oof'][name]:raise ValueError('Pooled metrics differ')
    if evaluate(y,baseline,.5)!=metrics['baseline_fixed_0_5']:raise ValueError('Baseline comparison differs')
    for number in range(1,6):
        selected=folds==number
        for name,t in [('fixed_0_5',.5),('inner_selected',thresholds[selected])]:
            if evaluate(y[selected],p[selected],t)!=metrics['per_fold'][str(number)][name]:raise ValueError('Fold metrics differ')
    return {'status':'reproduced_without_refitting','subjects':119,'fold_pipelines':35,
            'inner_pipelines':30,'outer_pipelines':5,'outer_predictions':119,
            'training_only_scaling_pca_verified':True,'candidate_and_threshold_decisions_reproduced':True}


def verify_run(output: Path, source_root: Path | None = None) -> dict[str, Any]:
    """Check exact artifact coverage before loading any private saved pipeline."""
    expected=json.loads((output/'checksum_manifest.json').read_text())
    actual=tree_checksums(output);actual.pop('checksum_manifest.json',None)
    if expected!=actual:raise ValueError('Run checksum or artifact coverage differs')
    # Match the extraction/training process native thread policy on fresh loads.
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        proof=verify_models(output)
    if source_root is not None:
        configuration=json.loads((output/'experiment_config.json').read_text())
        for path,digest in configuration['source_hashes'].items():checked_file(source_root,path,digest)
        for name,path in [('experiment1',Path(configuration['experiment1'])),('readiness',Path(configuration['readiness']))]:
            if tree_checksums(path)!=configuration['original_sha256'][name]:raise ValueError('Original experiment/readiness changed')
    return {**proof,'checksum_files':len(expected),'external_sources_checked':source_root is not None}


def plots(output: Path, y: np.ndarray, p: np.ndarray, baseline: np.ndarray, metrics: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(11,9))
    for scores,label in [(baseline,'Experiment 1 LAB'),(p,'Experiment 2 hybrid')]:
        false_positive,true_positive,_=roc_curve(y,scores)
        axes[0,0].plot(false_positive,true_positive,label=label)
        precision,recall,_=precision_recall_curve(y,scores)
        axes[0,1].plot(recall,precision,label=label)
    axes[0,0].plot([0,1],[0,1],'k--',alpha=.4)
    axes[0,0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Internal OOF ROC')
    axes[0,1].axhline(y.mean(),color='gray',linestyle='--',label='Cohort prevalence')
    axes[0,1].set(xlabel='Recall',ylabel='Precision',title='Internal OOF precision–recall')
    for ax in axes[0]:ax.legend();ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    for ax,name,title in [(axes[1,0],'fixed_0_5','Hybrid fixed threshold 0.5'),(axes[1,1],'inner_selected','Hybrid training-selected thresholds')]:
        matrix=np.array(metrics['pooled_oof'][name]['confusion_matrix_labels_0_1'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100)
        for (row,column),value in np.ndenumerate(matrix):ax.text(column,row,str(value),ha='center',va='center',color='white' if value>50 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-anemia','Anemia'],yticklabels=['Non-anemia','Anemia'],xlabel='Predicted',ylabel='Reference',title=title)
    figure.suptitle('119 subjects, 19 anemia; exploratory five-fold subject CV')
    figure.tight_layout();figure.savefig(output/'evaluation.png',dpi=180);figure.savefig(output/'evaluation.svg');plt.close(figure)


def report(output: Path, configuration: dict[str, Any], metrics: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines=['# Local manual-eyelid Experiments 1 and 2','',
           'Completed exploratory internal five-fold subject CV on the same 119 subjects (19 anemia/100 non-anemia). No external classifier holdout exists. Experiment 1 is unchanged.','',
           '| Model / decision rule | Architecture | Training | Accuracy | Sensitivity | Specificity | Balanced accuracy | Precision | F1 | ROC-AUC | AP |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    values=[('Experiment 1 / 0.5','Six LAB → logistic','Fixed five-fold',metrics['baseline_fixed_0_5']),
            ('Experiment 2 / 0.5','Six LAB + frozen CNN → training PCA8 → LightGBM','Nested 5×3 CV',metrics['pooled_oof']['fixed_0_5']),
            ('Experiment 2 / inner-selected','Same hybrid','Inner sensitivity at specificity ≥0.80',metrics['pooled_oof']['inner_selected'])]
    for name,architecture,training,value in values:
        formatted=['undefined' if value[k] is None else f'{value[k]:.6f}' for k in METRICS]
        lines.append('| '+' | '.join([name,architecture,training,*formatted])+' |')
    lines+=['','Matrices use reference rows and predicted columns ordered non-anemia/anemia.']
    for name,_,_,value in values:lines.append(f"- {name}: `{value['confusion_matrix_labels_0_1']}`")
    lines+=['','## Inner decisions','', '| Outer fold | Weighting | Inner AP | Threshold | Inner sensitivity | Inner specificity |', '| --- | --- | --- | --- | --- | --- |']
    for number,record in enumerate(records,1):
        lines.append(f"| {number} | {record['selected_candidate']} | {record['inner_average_precision'][record['selected_candidate']]:.6f} | {record['threshold']:.6f} | {record['inner_sensitivity']:.6f} | {record['inner_specificity']:.6f} |")
    lines+=['','Inner performance supports selection only. The specificity constraint applies to inner predictions, not guaranteed outer performance. Class weighting is fold-local and the scores are not calibrated.','',
            '## Paired uncertainty','', 'Intervals below compare hybrid minus fixed-threshold Experiment 1 on identical resampled OOF subjects.']
    for rule in ('fixed_0_5','inner_selected'):
        lines.append(f'\n### {rule}\n')
        uncertainty=metrics['uncertainty'][rule]
        for name in ('sensitivity','specificity','roc_auc','average_precision'):
            delta=uncertainty['paired_delta_exp02_minus_fixed_exp01'][name]
            lines.append(f"- {name}: delta {delta['estimate']:.6f}, conditional paired 95% interval {delta['interval']}")
        lines.append(f"- Sensitivity Wilson 95% interval: {uncertainty['intervals']['sensitivity']}")
    lines+=['','## Interpretation and limitations','',
            'Only 19 positive cases support this comparison. Use the paired intervals and fold-level variability when assessing numerical differences; this run cannot establish clinical validity or external generalization. The selected-threshold comparison changes the decision rule, so it is separate from the like-for-like fixed-0.5 comparison. All 119 subjects are development subjects. No model is selected or retuned using the outer results.',
            'Mixed uncalibrated historical TIFF rendering persists. Manual-mask shape, interpolation boundaries and letterbox proportions may contribute to embeddings. There is no embedding-only/PCA-only ablation, calibration study, independent test cohort or device/subgroup validation. The two-candidate selection and threshold selection share inner OOF scores; those selection scores may be optimistic. Conditional bootstraps fix prevalence and omit refit/overlapping-training uncertainty.',
            'The eight-PC compression and fixed shallow trees intentionally limit complexity, but this single small study cannot establish that extra neural features justify their cost. A consistent acquisition/color workflow, more positive cases, new untouched data and prespecified ablations should precede clinical interpretation. Public-anemia transfer is a separate future research decision and was not run here.','',
            '## Reproduction','', 'The run manifest, configuration, copied implementation/environment, frozen public weights, private feature cache, all 35 fitted pipelines, inner decisions, folds and OOF rows are checksum-covered. No patient image/ROI/mask was copied. From the Experiment 2 worktree with the recorded Python environment:',
            '```bash',f'python -m src.local_eyelid_hybrid_run verify --output "{output}"',f'python -m src.local_eyelid_hybrid_run verify --output "{output}" --source-root "{configuration["source_root"]}"','```',
            'Verification loads saved models and cached features, never fits. Source verification checks the original experiment and authoritative source hashes. To create a new run, use a distinct output directory and the saved configuration; never overwrite this run.']
    with (output/'report.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')


def implementation_state() -> dict[str, str]:
    root=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True):
        raise ValueError('Commit the tested implementation before the prespecified real run')
    return {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'branch':subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),
            'worktree':str(root)}


def run(source_root: Path, experiment1: Path, bundle: Path, output: Path, weights: Path) -> dict[str, Any]:
    source_root,experiment1,bundle=source_root.resolve(),experiment1.resolve(),bundle.resolve()
    if any(not p.is_relative_to(source_root/'data/processed') for p in (experiment1,bundle)):
        raise ValueError('Baseline inputs must reside under the explicit source root')
    output=private_output_path(source_root,output)
    state=implementation_state()
    from .local_eyelid_validation import source_snapshot
    canonical=source_root/'data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv'
    snapshot=source_snapshot(canonical,canonical.parent/'provenance.json')
    originals={'experiment1':tree_checksums(experiment1),'readiness':tree_checksums(bundle)}
    inputs=validated_inputs(source_root,experiment1,bundle)
    backbone=FrozenMobileNet(weights)
    source_hashes={str(canonical.relative_to(source_root)):snapshot['sha256'],str((canonical.parent/'provenance.json').relative_to(source_root)):snapshot['provenance_sha256']}
    baseline_config=json.loads((experiment1/'experiment_config.json').read_text())
    source_hashes.update(baseline_config['source_files'])
    for row in inputs['rows'].values():
        for key,digest in [('source_image_path','original_source_sha256'),('annotation_image_path','annotation_image_sha256'),('roi_path','roi_sha256'),('mask_path','mask_sha256'),('source_review_package_path','source_review_package_sha256')]:
            source_hashes[row[key]]=row[digest]
    code_root=Path(__file__).resolve().parents[1]
    code_files=sorted((code_root/'src').glob('*.py'))
    configuration={'experiment':'local_manual_eyelid_exp02_hybrid_v1','created_at':datetime.now(ZoneInfo('Asia/Manila')).isoformat(),
                   **state,'source_root':str(source_root),'experiment1':str(experiment1),'readiness':str(bundle),
                   'cohort':{'subjects':119,'anemia':19,'non_anemia':100,'holdout':'none'},
                   'canonical_source_snapshot_before_read':snapshot,'source_hashes':source_hashes,'original_sha256':originals,
                   'folds_sha256':sha256(experiment1/'subject_folds.csv'),'features_baseline_sha256':sha256(bundle/'features.csv'),
                   'preprocessing':PREPROCESSING,'design':DESIGN,'weights':backbone.metadata,
                   'code_sha256':{str(p.relative_to(code_root)):sha256(p) for p in code_files},
                   'approval':'Owner authorized Experiment 2 autonomous hybrid implementation, identical cohort/folds, nested selection and research thresholds; no clinical operating criterion approved.'}
    output.mkdir(mode=0o700)
    write_json(output/'experiment_config.json',configuration)
    write_json(output/'pretraining_verification.json',inputs['baseline_verification'])
    for name,path in [('subject_folds.csv',experiment1/'subject_folds.csv'),('baseline_features.csv',bundle/'features.csv'),('verified_eyelid_inputs.csv',bundle/'verified_eyelid_inputs.csv'),('duplicate_review.csv',experiment1/'duplicate_review_used.csv'),('imagenet_weights.pth',weights)]:
        copy_verified(path,output/name)
    for path in code_files:copy_verified(path,output/'implementation'/'src'/path.name)
    copy_verified(code_root/'requirements.txt',output/'implementation'/'requirements.txt')
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'packages':sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get('Name'))}
    write_json(output/'environment.json',environment)
    log=[{'event':'source_verification_passed','subjects':119,'anemia':19,'non_anemia':100}]
    with threadpool_limits(limits=1):
        x=extract_features(source_root,inputs,backbone)
        np.savez_compressed(output/'features.npz',subjects=np.array(inputs['subjects']),features=x)
        p=np.full(119,np.nan);thresholds=np.full(119,np.nan);records=[]
        for number in range(1,6):
            directory=output/f'fold_{number}';directory.mkdir(mode=0o700)
            training=inputs['folds']!=number;validation=inputs['folds']==number
            _,p[validation],record=fit_outer_training(x[training],inputs['labels'][training],
                [sid for sid,selected in zip(inputs['subjects'],training) if selected],x[validation],directory)
            thresholds[validation]=record['threshold'];records.append(record)
            write_json(directory/'selection.json',record)
            event={'event':'outer_fold_completed','fold':number,'selected_candidate':record['selected_candidate'],
                   'threshold':record['threshold'],'validation_subjects':int(validation.sum())}
            log.append(event);print(json.dumps(event),flush=True)
        if not np.isfinite(p).all() or not np.isfinite(thresholds).all():raise ValueError('Incomplete OOF coverage')
        baseline_rows={r['subject_id']:r for r in strict_csv(experiment1/'oof_predictions.csv')}
        baseline=np.array([float(baseline_rows[s]['lab_logistic_probability']) for s in inputs['subjects']])
        rows=[{'subject_id':s,'label':int(y),'fold':int(f),'hybrid_probability':float(prob),'fixed_threshold':.5,'fixed_prediction':int(prob>=.5),
               'selected_threshold':float(t),'selected_prediction':int(prob>=t),'baseline_probability':float(base)}
              for s,y,f,prob,t,base in zip(inputs['subjects'],inputs['labels'],inputs['folds'],p,thresholds,baseline)]
        write_csv(output/'oof_predictions.csv',rows)
        y=inputs['labels'];folds=inputs['folds']
        metrics={'status':'completed_nested_internal_subject_cv','subjects':119,'anemia':19,'non_anemia':100,
                 'baseline_fixed_0_5':evaluate(y,baseline,.5),
                 'pooled_oof':{'fixed_0_5':evaluate(y,p,.5),'inner_selected':evaluate(y,p,thresholds)},
                 'per_fold':{str(n):{'fixed_0_5':evaluate(y[folds==n],p[folds==n],.5),'inner_selected':evaluate(y[folds==n],p[folds==n],thresholds[folds==n])} for n in range(1,6)},
                 'uncertainty':{name:conditional_intervals(y,p,t,baseline) for name,t in [('fixed_0_5',.5),('inner_selected',thresholds)]},
                 'configuration_sha256':sha256(output/'experiment_config.json')}
        write_json(output/'metrics.json',metrics);plots(output,y,p,baseline,metrics);report(output,configuration,metrics,records)
        proof=verify_models(output)
    if originals!={'experiment1':tree_checksums(experiment1),'readiness':tree_checksums(bundle)}:
        raise ValueError('Original artifacts changed during Experiment 2')
    for path,digest in source_hashes.items():checked_file(source_root,path,digest)
    write_json(output/'verification.json',{**proof,'originals_unchanged':True,'source_hashes_rechecked':len(source_hashes)})
    log.append({'event':'completed_and_reproduced','outer_predictions':119,'pipelines':35})
    write_json(output/'training_log.json',log)
    write_json(output/'run_manifest.json',{'status':metrics['status'],'configuration_sha256':sha256(output/'experiment_config.json'),
               'git_commit':state['git_commit'],'output':str(output),'models':35,'originals_unchanged':True,
               'metrics_file':'metrics.json','report_file':'report.md','privacy':'Local owner-only ignored output; no patient images copied or external trackers used'})
    for path in output.rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
    write_json(output/'checksum_manifest.json',tree_checksums(output))
    result=verify_run(output)
    return {'status':metrics['status'],'output':str(output),'verification':result,'pooled_oof':metrics['pooled_oof']}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    trainer=sub.add_parser('run')
    for name in ('source-root','experiment1','bundle','output','weights'):trainer.add_argument(f'--{name}',type=Path,required=True)
    verifier=sub.add_parser('verify');verifier.add_argument('--output',type=Path,required=True);verifier.add_argument('--source-root',type=Path)
    args=parser.parse_args()
    result=run(args.source_root,args.experiment1,args.bundle,args.output,args.weights) if args.command=='run' else verify_run(args.output,args.source_root)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
