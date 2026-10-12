"""Run only approved-anatomy regional candidates; seal private artifacts."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .eyelid_color_run import write_json,prior_unchanged
from .eyelid_glare_prepare import EXP4
from .eyelid_glare_models import DESIGN,CANDIDATES,BY_ID,fit_outer
from .eyelid_glare_features import FEATURES
from .eyelid_glare_evaluation import compare,stability
from .local_eyelid_validation import sha256
from .preserve_local_experiment import tree_checksums


def registry(output: Path) -> dict:
    with np.load(output/'features.npz',allow_pickle=False) as cache: folds=cache['folds']
    with np.load(output/'candidate_predictions.npz',allow_pickle=False) as cache: probabilities={k:cache[k].copy() for k in cache.files}
    results={};selected=np.full(len(folds),np.nan);selected_t=selected.copy()
    for c in CANDIDATES:
        cutoff=np.full(len(folds),np.nan)
        for fold in range(1,6):
            record=json.loads((output/f'fold_{fold}/selection.json').read_text())
            cutoff[folds==fold]=record['candidates'][c['id']]['rule']['threshold']
            if record['selected']==c['id']:
                selected[folds==fold]=probabilities[c['id']][folds==fold];selected_t[folds==fold]=cutoff[folds==fold]
        results[c['id']+'_fixed']=(probabilities[c['id']],np.full(len(folds),.5))
        results[c['id']+'_selected']=(probabilities[c['id']],cutoff)
    if not np.isfinite(selected).all():raise ValueError('Incomplete selected OOF')
    results['nested_selected_fixed']=(selected,np.full(len(folds),.5));results['nested_selected']=(selected,selected_t)
    return results


def check_exp4_comparator(exp4: Path, folds: np.ndarray, result: tuple) -> None:
    """Compare exact binary/JSON evidence, avoiding CSV decimal parsing loss."""
    with np.load(exp4/'candidate_predictions.npz',allow_pickle=False) as cache:
        np.testing.assert_array_equal(result[0],cache['A_robust_lr_c1_none'])
    thresholds=np.full(len(folds),np.nan)
    for fold in range(1,6):
        saved=json.loads((exp4/f'fold_{fold}/selection.json').read_text())
        thresholds[folds==fold]=saved['candidates']['A_robust_lr_c1_none']['rules']['0.8']['threshold']
    np.testing.assert_array_equal(result[1],thresholds)


def run(root: Path, output: Path) -> dict:
    os.umask(0o077);start=time.monotonic();root=root.resolve();output=output.resolve()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_exp04b_glare_regional_'):
        raise ValueError('Require private Experiment4B preparation output')
    if (output/'run_manifest.json').exists() or (output/'fold_1').exists():raise FileExistsError('Never overwrite or resume a completed/attempted run')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')],cwd=root).returncode:raise ValueError('Output not ignored')
    config=json.loads((output/'preparation_config.json').read_text())
    prior_unchanged(root,config['previous_checksums'])
    for path,digest in config['source_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source changed before training')
    audit=json.loads((output/'glare_audit_summary.json').read_text())
    if audit['anatomical_regional_models']['state']!='READY_FOR_EXPLORATORY_TRAINING':raise ValueError('Anatomical input gate failed')
    with np.load(output/'features.npz',allow_pickle=False) as cache:
        subjects=cache['subjects'].tolist();y=cache['labels'];folds=cache['folds']
        features={k:cache['original_'+k] for k in FEATURES}
    old=pd.read_csv(root/'data/processed'/EXP4/'oof_predictions.csv',dtype={'subject_id':str}).set_index('subject_id').loc[subjects]
    np.testing.assert_array_equal(old.label,y);np.testing.assert_array_equal(old.fold,folds)
    repo=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    code={str(p.relative_to(repo)):sha256(p) for p in sorted((repo/'src').glob('*.py'))}
    for rel,digest in code.items():
        target=output/'implementation'/rel;target.parent.mkdir(exist_ok=True,mode=0o700);shutil.copy2(repo/rel,target)
        if sha256(target)!=digest:raise ValueError('Implementation copy mismatch')
    write_json(output/'experiment_config.json',{'design':DESIGN,'training_commit':commit,'code_sha256':code,
               'environment':{'python':platform.python_version(),'platform':platform.platform(),
                              'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','opencv-python','Pillow','joblib','threadpoolctl']},
                              'threads':1},'preparation_config_sha256':sha256(output/'preparation_config.json'),
               'input_feature_sha256':sha256(output/'features.npz'),'glare_models_run':False})
    probabilities={c['id']:np.full(len(y),np.nan) for c in CANDIDATES}
    for fold in range(1,6):
        tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        if set(tr)&set(va) or len(set(subjects))!=len(y):raise ValueError('Subject leakage')
        directory=output/f'fold_{fold}';directory.mkdir(mode=0o700)
        p,rec=fit_outer({k:v[tr] for k,v in features.items()},y[tr],[subjects[i] for i in tr],{k:v[va] for k,v in features.items()},directory)
        rec.update(outer_fold=fold,train_indices=tr.tolist(),validation_indices=va.tolist(),validation_subjects=[subjects[i] for i in va])
        write_json(directory/'selection.json',rec)
        for name,v in p.items():probabilities[name][va]=v
        print(f'Fitted outer fold {fold}: selected {rec["selected"]}',flush=True)
    np.savez_compressed(output/'candidate_predictions.npz',**probabilities)
    scores=registry(output)
    check_exp4_comparator(root/'data/processed'/EXP4,folds,scores['robust_none_selected'])
    scores['exp4_joint_primary']=(old.exp04_joint_selected_probability.to_numpy(),old.exp04_joint_selected_threshold.to_numpy())
    table=pd.DataFrame({'subject_id':subjects,'label':y,'fold':folds})
    for name,(p,t) in scores.items():table[name+'_probability']=p;table[name+'_threshold']=t
    table.to_csv(output/'oof_predictions_private.csv',index=False)
    metrics=compare(y,folds,scores);write_json(output/'metrics.json',metrics)
    write_json(output/'coefficient_stability.json',stability(output))
    plots(output,y,scores,metrics)
    report(output,metrics)
    prior_unchanged(root,config['previous_checksums'])
    for path,digest in config['source_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source changed during training')
    write_json(output/'run_manifest.json',{'status':'ANATOMICAL_REGIONAL_COMPLETE_GLARE_EXCLUSION_BLOCKED','subjects':len(y),'anemia':int(y.sum()),
              'classifier_pipelines':80,'training_commit':commit,'elapsed_training_evaluation_seconds':time.monotonic()-start,
              'unchanged_exp4_comparator_exact_decisions':True,'previous_experiments_unchanged':True,
              'glare_exclusion_models_run':False,'strict_verified_glare_subset_subjects':0})
    for path in output.rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
    write_json(output/'checksum_manifest.json',tree_checksums(output))
    return {'models':80,'status':'sealed','output':str(output)}


def plots(output: Path,y: np.ndarray,scores: dict,metrics: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    chosen=['robust_none_selected','regional_none_selected','regional_balanced_selected','nested_selected']
    for name in chosen:
        p,t=scores[name];fpr,tpr,_=roc_curve(y,p);prec,rec,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=name.replace('_selected',''));axes[1].plot(rec,prec)
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='OOF ROC');axes[0].legend(fontsize=8)
    axes[1].axhline(y.mean(),color='gray',linestyle='--');axes[1].set(xlabel='Recall',ylabel='Precision',title='OOF precision-recall')
    labels=[]
    for i,name in enumerate(chosen):
        m=metrics['models'][name];ci=m['intervals_95']['sensitivity'];axes[2].errorbar(m['sensitivity'],i,xerr=[[m['sensitivity']-ci[0]],[ci[1]-m['sensitivity']]],fmt='o')
        labels.append(f"{name.replace('_selected','')}\n{m['true_positives']}/19; specificity {m['specificity']:.0%}")
    axes[2].set(yticks=range(len(labels)),yticklabels=labels,xlim=(0,1),xlabel='Sensitivity with Wilson95%',title='Inner specificity constraint0.80')
    axes[2].tick_params(axis='y',labelsize=8)
    fig.suptitle('AIMA 4B - exploratory anatomy-only comparison; glare proposals excluded from training')
    fig.tight_layout();fig.savefig(output/'evaluation.png',dpi=180);fig.savefig(output/'evaluation.svg');plt.close(fig)


def report(output: Path,metrics: dict) -> None:
    lines=['# Experiment4B private comparison report','',
           'Unchanged119 subjects19/100; same five outer folds; four fixed robust18/regional22 unweighted/balanced C1 candidates. AP/model/cutoff decisions use only three inner folds. Glare proposals are diagnostic only: no verified glare records, no exclusion classifiers. All repeated-subject results exploratory.','',
           '| Model/rule | Sens. | Spec. | Bal.acc. | Accuracy | Precision | F1 | AUC | AP | TP/FN/TN/FP |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,m in metrics['models'].items():
        lines.append('| '+name+' | '+' | '.join(f'{m[k]:.4f}' if m[k] is not None else 'undefined' for k in ['sensitivity','specificity','balanced_accuracy','accuracy','precision','f1','roc_auc','average_precision'])+f" | {m['true_positives']}/{m['false_negatives']}/{m['true_negatives']}/{m['false_positives']} |")
    lines+=['','Full confusion matrices, per-fold metrics, Wilson/conditional intervals and paired differences are in metrics.json. Inner0.80 specificity is not an outer-performance guarantee. Original robust comparator matches sealed Experiment4 predictions/decisions. Coefficient signs/ranks are descriptive, with correlated features and overlapping training sets. Review all119 masks, including empty ones; retain pale tissue. No clinical/external-validation claim.']
    (output/'comparison_report_private.md').write_text('\n'.join(lines)+'\n')
    guide='''# Reproduction\n\nFrom the Experiment4B worktree with the project Python3.10 environment, one CPU thread and PYTHONPYCACHEPREFIX in a writable temporary directory:\n\n```bash\npython -m src.eyelid_glare_reproduce --source-root /Users/madz/Documents/aima-screening --output /Users/madz/Documents/aima-screening/data/processed/local_eyelid_exp04b_glare_regional_2026-10-12_v1 --fresh-features\n```\n\nChecksum verification occurs before model loading; all80 checkpoints, decisions, partitions and metrics are recomputed without fitting. Fresh mode re-generates all119 glare proposals from unchanged source pixels, compares masks and original/proposed-exclusion features exactly. Sources and Experiments1-4 are rechecked. Do not change the sealed review queue. Record future acceptance/corrections in a new versioned reviewed bundle, bound to proposal hashes; recheck class support in the verified subset before any exclusion classifier. No auto mask is approved by this run.\n'''
    guide=guide.replace('local_eyelid_exp04b_glare_regional_2026-10-12_v1',output.name)
    (output/'REPRODUCTION.md').write_text(guide)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):print(run(args.source_root,args.output))
