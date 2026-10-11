"""Exclusive private Experiment 4 execution, comparisons and artifact sealing."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .eyelid_color_data import PREPROCESSING, FEATURE_DEFINITIONS, FEATURE_SETS, extract_all
from .eyelid_color_models import DESIGN, CANDIDATES, SPECIFICITIES, fit_outer, SEED
from .eyelid_transfer_evaluation import compare_scores
from .eyelid_transfer_run import historical_predictions
from .local_eyelid_hybrid_features import validated_inputs
from .local_eyelid_hybrid_evaluation import evaluate
from .local_eyelid_validation import sha256, source_snapshot
from .preserve_local_experiment import tree_checksums, copy_verified

PRIOR_NAMES = ('local_eyelid_classification_2026-10-10_v1','local_eyelid_readiness_2026-10-10_v2',
               'local_eyelid_classification_exp02_hybrid_2026-10-11_v1','local_eyelid_classification_exp03_transfer_2026-10-11_v3')
ABLATIONS = [f'{p}_{f}_lr_c1_none' for p in ('A','B') for f in FEATURE_SETS]
ABLATIONS += ['A_robust_lr_c1_balanced','B_robust_lr_c1_balanced']


def write_json(path: Path, value: Any) -> None:
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def destination(source_root: Path, output: Path) -> Path:
    root, output = source_root.resolve(), output.resolve()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_classification_exp04_color_optimization_'):
        raise ValueError('Require distinct private Experiment 4 output')
    if output.exists(): raise FileExistsError('Never overwrite a completed or attempted experiment')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')],cwd=root).returncode:
        raise ValueError('Output must be Git-ignored')
    return output


def prior_unchanged(source_root: Path, expected: dict[str, Any]) -> None:
    for name, checksums in expected.items():
        if tree_checksums(source_root/'data/processed'/name) != checksums:
            raise ValueError('Historical experiment changed')


def score_registry(output: Path, source_root: Path, subjects: list[str], y: np.ndarray, folds: np.ndarray) -> dict:
    historical = historical_predictions(source_root,subjects,y,folds)
    exp3 = pd.read_csv(source_root/'data/processed'/PRIOR_NAMES[-1]/'oof_predictions.csv',dtype={'subject_id':str}).set_index('subject_id').loc[subjects]
    np.testing.assert_array_equal(exp3.label,y);np.testing.assert_array_equal(exp3.fold,folds)
    registry = {'exp01_fixed':(historical['exp01'],np.full(len(y),.5)),
                'exp02_fixed':(historical['exp02'],np.full(len(y),.5)),
                'exp02_selected':(historical['exp02'],historical['exp02_thresholds']),
                'lab_nested_selected':(exp3.lab_nested_selected_probability.to_numpy(),exp3.lab_nested_selected_threshold.to_numpy()),
                'exp03_selected':(exp3.selected_selected_probability.to_numpy(),exp3.selected_selected_threshold.to_numpy())}
    np.testing.assert_allclose(registry['lab_nested_selected'][0],historical['exp01'],rtol=0,atol=1e-12)
    with np.load(output/'candidate_predictions.npz',allow_pickle=False) as cache:
        for policy in ('A_selected','B_selected','joint_selected'):
            p=np.full(len(y),np.nan);rules={str(s):np.full(len(y),np.nan) for s in SPECIFICITIES}
            for fold in range(1,6):
                info=json.loads((output/f'fold_{fold}/selection.json').read_text());c=info['selected_policies'][policy];idx=folds==fold
                p[idx]=cache[c][idx]
                for s in SPECIFICITIES: rules[str(s)][idx]=info['candidates'][c]['rules'][str(s)]['threshold']
            registry[f'exp04_{policy}_fixed']=(p,np.full(len(y),.5))
            registry[f'exp04_{policy}']=(p,rules['0.8'])
            if policy=='joint_selected':
                for s in (.7,.9):registry[f'exp04_joint_specificity_{s}']=(p,rules[str(s)])
        for c in ABLATIONS:
            t=np.full(len(y),np.nan)
            for fold in range(1,6):
                info=json.loads((output/f'fold_{fold}/selection.json').read_text());t[folds==fold]=info['candidates'][c]['rules']['0.8']['threshold']
            registry[f'ablation_{c}']=(cache[c].copy(),t)
    return registry


def regression(output: Path, features: np.ndarray, hb: np.ndarray, sex: np.ndarray, y: np.ndarray, folds: np.ndarray) -> dict:
    """Optional fixed Ridge feasibility check after primary classification."""
    directory=output/'regression';directory.mkdir(mode=0o700)
    p=np.full(len(hb),np.nan);mean=np.full(len(hb),np.nan)
    for fold in range(1,6):
        tr,va=folds!=fold,folds==fold
        model=Pipeline([('scaler',StandardScaler()),('ridge',Ridge(alpha=10.0))]).fit(features[tr],hb[tr])
        baseline=DummyRegressor(strategy='mean').fit(features[tr],hb[tr])
        p[va]=model.predict(features[va]);mean[va]=baseline.predict(features[va])
        joblib.dump(model,directory/f'ridge_fold_{fold}.joblib',compress=3)
        joblib.dump(baseline,directory/f'mean_fold_{fold}.joblib',compress=3)
    results={}
    for name,prob in [('ridge',p),('training_mean',mean)]:
        rng=np.random.default_rng(SEED);error=np.abs(hb-prob)
        ci=np.quantile([error[rng.integers(0,len(hb),len(hb))].mean() for _ in range(2000)],[.025,.975]).tolist()
        score=expit(np.where(sex=='M',13.,12.)-prob)
        results[name]={'mae_g_dl':float(mean_absolute_error(hb,prob)),'rmse_g_dl':float(np.sqrt(mean_squared_error(hb,prob))),
                       'r2':float(r2_score(hb,prob)),'mae_conditional_95_interval':ci,
                       'anemia_information':evaluate(y,score,np.nextafter(.5,np.inf)),
                       'per_fold':{str(f):{'mae_g_dl':float(mean_absolute_error(hb[folds==f],prob[folds==f])),
                                           'rmse_g_dl':float(np.sqrt(mean_squared_error(hb[folds==f],prob[folds==f])))} for f in range(1,6)}}
    results['limitations']='Fixed alpha10, managed robust18, training-only scaling; no tuning. Conditional Hb error bootstrap omits refit uncertainty. Sex only supplies existing anemia cutoffs for evaluation, never a regression predictor. Sigmoid(cutoff-predicted Hb) is a ranking link, not calibrated probability. No clinical Hb measurement claim.'
    np.savez_compressed(directory/'predictions.npz',ridge=p,training_mean=mean)
    write_json(directory/'metrics.json',results)
    return results


def figures(output: Path, y: np.ndarray, registry: dict, metrics: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for name,label in [('lab_nested_selected','Historical LAB'),('exp02_selected','Hybrid'),('exp03_selected','Transfer'),('exp04_joint_selected','Exp4 joint selected')]:
        p,t=registry[name];fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        axes[0,0].plot(fpr,tpr,label=label);axes[0,1].plot(recall,precision,label=label)
    axes[0,0].plot([0,1],[0,1],'k--',alpha=.4);axes[0,1].axhline(y.mean(),color='gray',linestyle='--')
    axes[0,0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Internal OOF ROC')
    axes[0,1].set(xlabel='Recall',ylabel='Precision',title='Internal OOF precision–recall')
    for ax in axes[0]:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend()
    for ax,name,title in [(axes[1,0],'exp04_joint_selected_fixed','Joint selected / cutoff 0.5'),(axes[1,1],'exp04_joint_selected','Joint selected / inner specificity ≥0.80')]:
        matrix=np.array(metrics['models'][name]['confusion_matrix_labels_0_1']);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100)
        for (r,c),v in np.ndenumerate(matrix):ax.text(c,r,str(v),ha='center',va='center',color='white' if v>50 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-anemia','Anemia'],yticklabels=['Non-anemia','Anemia'],xlabel='Predicted',ylabel='Reference',title=title)
    fig.suptitle('Exploratory development: 119 subjects, 19 anemia, unchanged five outer folds');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/f'evaluation.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,label in [('exp04_joint_selected_fixed','Fixed .5'),('exp04_joint_specificity_0.7','Inner specificity≥.70'),('exp04_joint_selected','Inner specificity≥.80'),('exp04_joint_specificity_0.9','Inner specificity≥.90')]:
        m=metrics['models'][name];ax.scatter(m['specificity'],m['sensitivity'],label=f"{label}: {m['true_positives']}/19 TP, {m['false_positives']}/100 FP")
    ax.set(xlim=(0,1.02),ylim=(0,1.02),xlabel='Outer specificity actually achieved',ylabel='Outer sensitivity',title='Predefined rules; no rule selected from these outer results');ax.legend();fig.tight_layout()
    fig.savefig(output/'tradeoffs.png',dpi=180);fig.savefig(output/'tradeoffs.svg');plt.close(fig)


def report(output: Path, metrics: dict, records: list[dict], candidate_metrics: dict, regression_metrics: dict) -> None:
    lines=['# Experiment 4 — verified exploratory local color classification','',
           'Same 119 subjects (19 anemia/100 non-anemia), approved manual anatomy and five outer folds. Prior experiments informed this design, so results are development evidence, not independent confirmation. Configurations and thresholds were selected inside outer training only.','',
           '| Model / rule | Sensitivity | Specificity | Balanced acc. | ROC-AUC | AP | TP | FN | TN | FP |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    primary=['exp01_fixed','lab_nested_selected','exp02_selected','exp03_selected','exp04_A_selected','exp04_B_selected','exp04_joint_selected_fixed','exp04_joint_selected','exp04_joint_specificity_0.7','exp04_joint_specificity_0.9']
    for name in primary:
        m=metrics['models'][name];lines.append('| '+name+' | '+' | '.join(f"{m[k]:.6f}" for k in ['sensitivity','specificity','balanced_accuracy','roc_auc','average_precision'])+' | '+' | '.join(str(m[k]) for k in ['true_positives','false_negatives','true_negatives','false_positives'])+' |')
    lines+=['','Full accuracy/precision/F1, Wilson and conditional bootstrap intervals, per-fold metrics and paired differences are in metrics.json.','',
            '## Frozen selections','', '| Fold | A policy | B policy | Joint policy | Joint .80 cutoff |','| --- | --- | --- | --- | ---: |']
    for i,r in enumerate(records,1):
        c=r['selected_policies']['joint_selected'];lines.append(f"| {i} | {r['selected_policies']['A_selected']} | {r['selected_policies']['B_selected']} | {c} | {r['candidates'][c]['rules']['0.8']['threshold']:.6f} |")
    lines+=['','## Controlled descriptive comparisons','', 'Every frozen candidate is refit and evaluated on the same outer folds. candidate_metrics.json and candidate_results.csv contain all 36 candidates at fixed .5 and inner-selected .80 cutoffs. Compare A/B with identical features/model to isolate rendering; compare original/robust/regional under the same model to examine features; compare none/balanced at the same rendering/features/C for weighting. No outer candidate result chooses the primary model. Ablations are exploratory and correlated; no causal attribution follows from a small difference.','',
            '## Diagnostics and rendering','', 'Private diagnostics/diagnostic_report_private.md identifies the twelve original missed cases by separate pseudonyms. Eight are mild and four moderate; median Hb11.55 versus10.8 g/dL in seven detected cases. Native focus medians are nearly identical, which does not support a blur-only explanation. Eleven TIFFs have pixel-identical Display P3 DNG previews under misleading sRGB tags; none is among the twelve original misses. Correcting these profiles is scientifically necessary but cannot directly change those twelve input images. It can change training distributions. Export tone curves, highlight clipping and absent capture color references remain unresolved. No glare or difficult subject was excluded.','',
            '## Optional Hb regression (after classification)','',f"Fixed managed robust18 Ridge(alpha10): MAE {regression_metrics['ridge']['mae_g_dl']:.6f} g/dL, RMSE {regression_metrics['ridge']['rmse_g_dl']:.6f}, R² {regression_metrics['ridge']['r2']:.6f}. Training-mean MAE {regression_metrics['training_mean']['mae_g_dl']:.6f}, R² {regression_metrics['training_mean']['r2']:.6f}. This feasibility output is separate from classification and not an Hb measurement.",'',
            '## Limits and reproduction','', 'Only19 positives: one additional detection changes sensitivity by5.3 percentage points. Intervals condition on saved subject scores, fix prevalence and omit refit/overlapping-fold uncertainty. Inner specificity constraints do not guarantee the same outer specificity. The same subjects have informed four experiments; no numeric increase establishes an independent improvement or clinical validity.','', 'REPRODUCTION.md explains checksum-first prediction/selection verification without fitting, plus fresh source-feature verification. All historical checksums and preservation references remain unchanged. Fitted models, patient rows, source linkage and images stay local and Git-ignored.']
    with (output/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')


def run(source_root: Path, preflight: Path, output: Path) -> dict:
    os.umask(0o077);cv2.setNumThreads(1);source_root=source_root.resolve();output=destination(source_root,output)
    code_root=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain'],cwd=code_root,text=True):raise ValueError('Freeze tested code in a clean commit before full training')
    previous=json.loads((preflight/'previous_checksums.json').read_text());prior_unchanged(source_root,previous)
    canonical=source_root/'data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv';snapshot=source_snapshot(canonical,canonical.parent/'provenance.json')
    fresh=json.loads((preflight/'fresh_local_audit.json').read_text())
    if fresh['blockers'] or snapshot['sha256']!=fresh['metadata_source']['sha256']:raise ValueError('Fresh input audit unavailable/stale')
    inputs=validated_inputs(source_root,source_root/'data/processed'/PRIOR_NAMES[0],source_root/'data/processed'/PRIOR_NAMES[1])
    subjects,y,folds=inputs['subjects'],inputs['labels'],inputs['folds']
    clinical=pd.DataFrame(json.loads((preflight/'fresh_eligibility.json').read_text())).set_index('subject_id').loc[subjects]
    np.testing.assert_array_equal(clinical.derived_anemia_label.astype(int),y)
    old=pd.read_csv(source_root/'data/processed'/PRIOR_NAMES[1]/'all_subject_eligibility.csv',dtype={'subject_id':str}).set_index('subject_id').loc[subjects]
    np.testing.assert_array_equal(clinical.normalized_hb_g_per_dl.astype(float),old.normalized_hb_g_per_dl.astype(float))
    hb=clinical.normalized_hb_g_per_dl.astype(float).to_numpy();sex=clinical.sex.to_numpy()
    files=sorted(Path(__file__).parent.glob('*.py'))
    source_hashes={str(canonical):snapshot['sha256'],str(canonical.parent/'provenance.json'):snapshot['provenance_sha256']}
    source_hashes.update(fresh['source_files'])
    for row in inputs['rows'].values():
        for path_key,hash_key in [('source_image_path','original_source_sha256'),('annotation_image_path','annotation_image_sha256'),('roi_path','roi_sha256'),('mask_path','mask_sha256'),('source_review_package_path','source_review_package_sha256')]:source_hashes[str(source_root/row[path_key])]=row[hash_key]
    config={'experiment':'local_manual_eyelid_exp04_color_optimization_v1','client_task_date':'2026-10-12',
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip(),
            'branch':subprocess.check_output(['git','branch','--show-current'],cwd=code_root,text=True).strip(),
            'worktree':str(code_root),'source_root':str(source_root),'design':DESIGN,'preprocessing':PREPROCESSING,
            'feature_names':{k:list(v) for k,v in FEATURE_SETS.items()},'feature_definitions':FEATURE_DEFINITIONS,
            'cohort':{'subjects':119,'anemia':19,'non_anemia':100,'holdout':'none; owner-selected all119 for development'},
            'canonical_source_snapshot_before_read':snapshot,'source_hashes':source_hashes,
            'code_sha256':{str(p.relative_to(code_root)):sha256(p) for p in files},'previous_checksums':previous,
            'preservation_refs':{ref:subprocess.check_output(['git','rev-parse',f'{ref}^{{commit}}'],cwd=source_root,text=True).strip() for ref in ['archive/eyelid-exp01-cielab-baseline','eyelid-exp01-cielab-baseline-2026-10-11']},
            'approval':'Owner attachment explicitly authorizes Experiment4 color/features/classical models/nested CV/full training/feature commits; labels,masks,membership and outer folds unchanged.'}
    output.mkdir(mode=0o700);write_json(output/'experiment_config.json',config)
    write_json(output/'readiness.json',{'state':'READY_FOR_EXPLORATORY_TRAINING','fresh_audit':fresh,'previous_reproduction':json.loads((preflight/'previous_reproduction.json').read_text())})
    for p in preflight.iterdir():
        if p.is_file():copy_verified(p,output/'diagnostics'/p.name)
    for p in files:copy_verified(p,output/'implementation'/p.relative_to(code_root))
    copy_verified(code_root/'requirements.txt',output/'implementation/requirements.txt')
    write_json(output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'native_threads':1,
                 'versions':{n:importlib.metadata.version(n) for n in ['numpy','Pillow','opencv-python','scikit-learn','scipy','lightgbm','joblib','matplotlib','torch','torchvision']}})
    pd.DataFrame({'subject_id':subjects,'label':y,'fold':folds}).to_csv(output/'subject_folds.csv',index=False)
    started=time.monotonic()
    with threadpool_limits(limits=1):
        features,audits=extract_all(source_root,inputs);write_json(output/'rendering_audit_private.json',audits)
        np.savez_compressed(output/'features.npz',**features,subjects=np.array(subjects),labels=y,folds=folds,hb_g_dl=hb,sex=sex)
        candidate_probabilities={c['id']:np.full(len(y),np.nan) for c in CANDIDATES};records=[]
        for f in range(1,6):
            tr,va=np.flatnonzero(folds!=f),np.flatnonzero(folds==f);directory=output/f'fold_{f}';directory.mkdir(mode=0o700)
            probabilities,record=fit_outer({k:x[tr] for k,x in features.items()},y[tr],[subjects[i] for i in tr],{k:x[va] for k,x in features.items()},directory)
            record['validation_subjects']=[subjects[i] for i in va];records.append(record);write_json(directory/'selection.json',record)
            for c,p in probabilities.items():candidate_probabilities[c][va]=p
            print(json.dumps({'outer_fold':f,'selected':record['selected_policies']}),flush=True)
        if not all(np.isfinite(v).all() for v in candidate_probabilities.values()):raise ValueError('Missing outer predictions')
        np.savez_compressed(output/'candidate_predictions.npz',**candidate_probabilities)
        candidate_metrics={};rows=[]
        for c in CANDIDATES:
            t=np.array([records[int(f)-1]['candidates'][c['id']]['rules']['0.8']['threshold'] for f in folds]);p=candidate_probabilities[c['id']]
            candidate_metrics[c['id']]={rule:{'pooled':evaluate(y,p,threshold),'per_fold':{str(f):evaluate(y[folds==f],p[folds==f],np.broadcast_to(threshold,y.shape)[folds==f]) for f in range(1,6)}} for rule,threshold in [('fixed',.5),('selected',t)]}
            rows.append({'candidate':c['id'],**candidate_metrics[c['id']]['selected']['pooled']})
        write_json(output/'candidate_metrics.json',candidate_metrics);pd.DataFrame(rows).to_csv(output/'candidate_results.csv',index=False)
        np.testing.assert_allclose(candidate_probabilities['A_original_lr_c1_none'],historical_predictions(source_root,subjects,y,folds)['exp01'],rtol=0,atol=1e-12)
        registry=score_registry(output,source_root,subjects,y,folds)
        metrics=compare_scores(y,folds,registry,replicates=2000,seed=SEED);write_json(output/'metrics.json',metrics)
        frame=pd.DataFrame({'subject_id':subjects,'label':y,'fold':folds})
        for name,(p,t) in registry.items():frame[f'{name}_probability']=p;frame[f'{name}_threshold']=t
        frame.to_csv(output/'oof_predictions.csv',index=False);figures(output,y,registry,metrics)
        print('Primary classification complete; running separate fixed Hb feasibility check',flush=True)
        reg=regression(output,features['B_robust'],hb,sex,y,folds);report(output,metrics,records,candidate_metrics,reg)
    prior_unchanged(source_root,previous)
    instructions=f'''# Saved-model reproduction, no training\n\nFrom the recorded Experiment4 implementation commit using its environment:\n\n```sh\nPYTHONPYCACHEPREFIX=/private/tmp/aima-exp04-pycache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\\n/Users/madz/Documents/aima-screening/aima_env/bin/python -m src.eyelid_color_reproduce \\\n --source-root {source_root} --output {output} --fresh-features\n```\n\nChecksums are verified before model deserialization. Verification never fits. It reproduces all720 classification and10 regression pipelines, inner selections/thresholds, allOOF scores and all pooled/per-fold metrics. Fresh-feature mode recalculates every source-based candidate vector. Original experiments and preservation refs are checked. New training requires a distinct versioned output, never this directory. Copied source/config/environment and the implementation commit support reproduction. Patient artifacts remain private and ignored.\n'''
    with (output/'REPRODUCTION.md').open('x') as f:f.write(instructions)
    manifest={'status':'training_and_evaluation_complete','classification_pipelines':720,'regression_pipelines':10,'subjects':119,'anemia':19,
              'elapsed_seconds_features_training_evaluation':time.monotonic()-started,'training_commit':config['git_commit'],
              'previous_experiments_unchanged':True,'output':str(output),'client_completion_date':'2026-10-12'}
    write_json(output/'run_manifest.json',manifest)
    for p in output.rglob('*'):p.chmod(0o700 if p.is_dir() else 0o600)
    write_json(output/'checksum_manifest.json',tree_checksums(output));return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.source_root,args.preflight,args.output),indent=2))

if __name__=='__main__':main()
