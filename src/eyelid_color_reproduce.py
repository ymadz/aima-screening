"""Checksum-first independent saved-model verification; never fits a model."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score,accuracy_score,f1_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from .eyelid_color_models import CANDIDATES,DESIGN,SPECIFICITIES,BY_ID,select,threshold_rule,SEED
from .eyelid_color_data import FEATURE_SETS,PREPROCESSING,extract_all
from .eyelid_color_run import PRIOR_NAMES,score_registry,prior_unchanged
from .local_eyelid_hybrid_features import validated_inputs
from .local_eyelid_hybrid_evaluation import evaluate
from .local_eyelid_validation import sha256,source_snapshot
from .preserve_local_experiment import tree_checksums


def check_model(model, candidate: dict, train_x: np.ndarray) -> None:
    params=model['classifier'].get_params()
    if params['class_weight'] != candidate['class_weight']:raise ValueError('Training class weighting differs')
    if candidate['family']=='logistic':
        if params['C']!=candidate['C'] or params['solver']!='lbfgs' or params['max_iter']!=2000 or params['penalty']!='l2':raise ValueError('Logistic settings differ')
        np.testing.assert_allclose(model['scaler'].mean_,train_x.mean(0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model['scaler'].var_,train_x.var(0),rtol=0,atol=1e-12)
        assert int(model['scaler'].n_samples_seen_)==len(train_x)
    else:
        if any(params[k]!=v for k,v in DESIGN['lightgbm_parameters'].items()):raise ValueError('LightGBM settings differ')
        assert model['columns'].names==FEATURE_SETS[candidate['features']]


def check_metrics(y: np.ndarray, p: np.ndarray, t: np.ndarray | float, saved: dict) -> None:
    measured=evaluate(y,p,t)
    for k,v in measured.items():
        if k!='research_threshold' and saved[k]!=v:raise ValueError(f'Metric differs: {k}')
    # Independently recompute ranking/class metrics using sklearn, not the report utility.
    for k,v in {'accuracy':accuracy_score(y,p>=t),'f1':f1_score(y,p>=t,zero_division=0),
                'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p)}.items():
        np.testing.assert_allclose(saved[k],v,rtol=0,atol=1e-14)


def reproduce(output: Path, source_root: Path, fresh_features: bool=False) -> dict:
    cv2.setNumThreads(1);output=output.resolve();source_root=source_root.resolve()
    expected=json.loads((output/'checksum_manifest.json').read_text());actual=tree_checksums(output);actual.pop('checksum_manifest.json',None)
    if actual!=expected:raise ValueError('Checksum coverage/content mismatch; no model loaded')
    config=json.loads((output/'experiment_config.json').read_text())
    if config['design']!=DESIGN or config['preprocessing']!=PREPROCESSING:raise ValueError('Frozen design differs')
    canonical=source_root/'data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv';snapshot=source_snapshot(canonical,canonical.parent/'provenance.json')
    if snapshot['sha256']!=config['canonical_source_snapshot_before_read']['sha256']:raise ValueError('Canonical source changed')
    for path,digest in config['source_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source bytes changed')
    for path,digest in config['code_sha256'].items():
        if sha256(output/'implementation'/path)!=digest or sha256(Path(__file__).resolve().parents[1]/path)!=digest:raise ValueError('Use recorded implementation for reproduction')
    prior_unchanged(source_root,config['previous_checksums'])
    for ref,commit in config['preservation_refs'].items():
        if subprocess.check_output(['git','rev-parse',f'{ref}^{{commit}}'],cwd=source_root,text=True).strip()!=commit:raise ValueError('Preservation reference changed')
    inputs=validated_inputs(source_root,source_root/'data/processed'/PRIOR_NAMES[0],source_root/'data/processed'/PRIOR_NAMES[1])
    with np.load(output/'features.npz',allow_pickle=False) as cache:
        subjects=cache['subjects'].tolist();y=cache['labels'];folds=cache['folds'];hb=cache['hb_g_dl'];sex=cache['sex']
        features={f'{p}_{f}':cache[f'{p}_{f}'] for p in ('A','B') for f in FEATURE_SETS}
    if subjects!=inputs['subjects']:raise ValueError('Cohort subject order changed')
    np.testing.assert_array_equal(y,inputs['labels']);np.testing.assert_array_equal(folds,inputs['folds']);np.testing.assert_allclose(features['A_original'],inputs['lab'],rtol=0,atol=1e-14)
    clinical=pd.DataFrame(json.loads((output/'diagnostics/fresh_eligibility.json').read_text())).set_index('subject_id').loc[subjects]
    np.testing.assert_array_equal(hb,clinical.normalized_hb_g_per_dl.astype(float));np.testing.assert_array_equal(sex,clinical.sex)
    predictions=np.load(output/'candidate_predictions.npz',allow_pickle=False);all_metrics=json.loads((output/'candidate_metrics.json').read_text());count=0
    for fold in range(1,6):
        tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold);ids=[subjects[i] for i in tr];inner_y=y[tr]
        record=json.loads((output/f'fold_{fold}/selection.json').read_text())
        if record['training_subjects']!=ids or record['training_labels']!=inner_y.tolist() or record['validation_subjects']!=[subjects[i] for i in va]:raise ValueError('Outer membership differs')
        scores={};splits=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(ids,inner_y))
        for c in CANDIDATES:
            x=features[c['key']];inner_p=np.full(len(tr),np.nan)
            for i,(tt,vv) in enumerate(splits,1):
                part=record['inner_partitions'][i-1]
                if part['train_indices']!=tt.tolist() or part['validation_indices']!=vv.tolist() or part['training_subjects']!=[ids[j] for j in tt] or part['validation_subjects']!=[ids[j] for j in vv]:raise ValueError('Inner subject scope differs')
                model=joblib.load(output/f'fold_{fold}'/c['id']/f'inner_{i}.joblib');check_model(model,c,x[tr[tt]])
                inner_p[vv]=model.predict_proba(x[tr[vv]])[:,1];count+=1
            detail=record['candidates'][c['id']];np.testing.assert_allclose(inner_p,detail['inner_probabilities'],rtol=0,atol=1e-14)
            scores[c['id']]=float(average_precision_score(inner_y,inner_p))
            if scores[c['id']]!=detail['inner_ap']:raise ValueError('Inner AP differs')
            for s in SPECIFICITIES:
                if threshold_rule(inner_y,inner_p,s)!=detail['rules'][str(s)]:raise ValueError('Inner threshold differs')
            outer=joblib.load(output/f'fold_{fold}'/c['id']/'outer.joblib');check_model(outer,c,x[tr]);count+=1
            np.testing.assert_allclose(outer.predict_proba(x[va])[:,1],predictions[c['id']][va],rtol=0,atol=1e-14)
            for rule,t in [('fixed',.5),('selected',detail['rules']['0.8']['threshold'])]:check_metrics(y[va],predictions[c['id']][va],t,all_metrics[c['id']][rule]['per_fold'][str(fold)])
        for p in ('A','B'):
            if select(scores,p)!=record['selected_policies'][f'{p}_selected']:raise ValueError('Render policy differs')
        if select(scores)!=record['selected_policies']['joint_selected']:raise ValueError('Joint policy differs')
    for c in CANDIDATES:
        thresholds=np.array([json.loads((output/f'fold_{f}/selection.json').read_text())['candidates'][c['id']]['rules']['0.8']['threshold'] for f in folds])
        for rule,t in [('fixed',.5),('selected',thresholds)]:check_metrics(y,predictions[c['id']],t,all_metrics[c['id']][rule]['pooled'])
    registry=score_registry(output,source_root,subjects,y,folds);metrics=json.loads((output/'metrics.json').read_text())
    frame=pd.read_csv(output/'oof_predictions.csv',dtype={'subject_id':str});assert frame.subject_id.tolist()==subjects
    np.testing.assert_array_equal(frame.label,y);np.testing.assert_array_equal(frame.fold,folds)
    for name,(p,t) in registry.items():
        np.testing.assert_allclose(frame[f'{name}_probability'],p,rtol=0,atol=1e-14);np.testing.assert_allclose(frame[f'{name}_threshold'],t,rtol=0,atol=1e-14)
        check_metrics(y,p,t,metrics['models'][name])
        for f in range(1,6):idx=folds==f;check_metrics(y[idx],p[idx],np.broadcast_to(t,y.shape)[idx],metrics['models'][name]['per_fold'][str(f)])
    reg=np.load(output/'regression/predictions.npz',allow_pickle=False);regmetrics=json.loads((output/'regression/metrics.json').read_text())
    for f in range(1,6):
        tr,va=folds!=f,folds==f
        for name in ('ridge','mean'):
            model=joblib.load(output/'regression'/f'{name}_fold_{f}.joblib');key='ridge' if name=='ridge' else 'training_mean'
            np.testing.assert_allclose(model.predict(features['B_robust'][va]),reg[key][va],rtol=0,atol=1e-14)
            if name=='ridge':
                assert model['ridge'].alpha==10.;np.testing.assert_allclose(model['scaler'].mean_,features['B_robust'][tr].mean(0),rtol=0,atol=1e-12)
                np.testing.assert_allclose(model['scaler'].var_,features['B_robust'][tr].var(0),rtol=0,atol=1e-12)
            else:np.testing.assert_allclose(model.constant_,hb[tr].mean(),rtol=0,atol=1e-14)
    for key in ('ridge','training_mean'):
        p=reg[key];measured={'mae_g_dl':mean_absolute_error(hb,p),'rmse_g_dl':np.sqrt(mean_squared_error(hb,p)),'r2':r2_score(hb,p)}
        for name,value in measured.items():np.testing.assert_allclose(regmetrics[key][name],value,rtol=0,atol=1e-14)
        check_metrics(y,expit(np.where(sex=='M',13.,12.)-p),np.nextafter(.5,np.inf),regmetrics[key]['anemia_information'])
    if fresh_features:
        fresh,audit=extract_all(source_root,inputs)
        for k in features:np.testing.assert_array_equal(fresh[k],features[k])
        if audit!=json.loads((output/'rendering_audit_private.json').read_text()):raise ValueError('Rendering audit differs')
    if tree_checksums(output)!=actual|{'checksum_manifest.json':sha256(output/'checksum_manifest.json')}:raise ValueError('Artifacts changed during verification')
    return {'status':'reproduced_without_refitting','subjects':len(y),'anemia':int(y.sum()),'classification_pipelines':count,
            'regression_pipelines':10,'checksum_files':len(expected),'inner_selections_thresholds_and_all_metrics_verified':True,
            'all_source_features_fresh_exact':fresh_features,'prior_experiments_and_refs_unchanged':True}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--fresh-features',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=1):result=reproduce(args.output,args.source_root,args.fresh_features)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
