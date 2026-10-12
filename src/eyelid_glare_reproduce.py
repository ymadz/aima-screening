"""Checksum-first saved-model and fresh-source reproduction without fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from .eyelid_color_models import BY_ID as OLD_MODELS,threshold_rule
from .eyelid_color_run import prior_unchanged
from .eyelid_color_verify import check_model,check_metrics
from .eyelid_glare_features import FEATURES,extract
from .eyelid_glare_masks import propose_glare
from .eyelid_glare_models import CANDIDATES,DESIGN,select
from .eyelid_glare_run import registry
from .eyelid_glare_evaluation import compare,stability
from .local_eyelid_validation import sha256,load_manual_pair
from .preserve_local_experiment import tree_checksums
from PIL import Image


def reproduce(output: Path, root: Path, fresh_features: bool=False) -> dict:
    cv2.setNumThreads(1);output=output.resolve();root=root.resolve()
    expected=json.loads((output/'checksum_manifest.json').read_text());actual=tree_checksums(output);actual.pop('checksum_manifest.json',None)
    if expected!=actual:raise ValueError('Checksum content/coverage mismatch; refuse deserialization')
    preparation=json.loads((output/'preparation_config.json').read_text());config=json.loads((output/'experiment_config.json').read_text())
    if config['design']!=DESIGN or sha256(output/'preparation_config.json')!=config['preparation_config_sha256']:
        raise ValueError('Frozen design or provenance differs')
    if sha256(output/'features.npz')!=config['input_feature_sha256']:raise ValueError('Feature cache changed')
    for path,digest in preparation['source_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source bytes changed')
    prior_unchanged(root,preparation['previous_checksums'])
    current=Path(__file__).resolve().parents[1]
    for path,digest in config['code_sha256'].items():
        if sha256(output/'implementation'/path)!=digest or sha256(current/path)!=digest:raise ValueError('Recorded implementation differs')
        saved=subprocess.check_output(['git','show',f"{config['training_commit']}:{path}"],cwd=root)
        if hashlib.sha256(saved).hexdigest()!=digest:raise ValueError('Recorded Git implementation differs')
    with np.load(output/'features.npz',allow_pickle=False) as cache:
        subjects=cache['subjects'].tolist();y=cache['labels'];folds=cache['folds'];features={k:cache['original_'+k] for k in FEATURES}
        all_features={k:cache[k] for k in cache.files if k.startswith(('original_','proposed_excluded_'))}
    if len(subjects)!=119 or len(set(subjects))!=119 or int(y.sum())!=19 or set(folds)!={1,2,3,4,5}:raise ValueError('Cohort differs')
    with np.load(output/'candidate_predictions.npz',allow_pickle=False) as cache:probabilities={k:cache[k] for k in cache.files}
    verified=0
    for fold in range(1,6):
        rec=json.loads((output/f'fold_{fold}/selection.json').read_text());tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        assert rec['train_indices']==tr.tolist() and rec['validation_indices']==va.tolist()
        assert rec['training_subjects']==[subjects[i] for i in tr] and rec['validation_subjects']==[subjects[i] for i in va]
        assert set(rec['training_subjects']).isdisjoint(rec['validation_subjects'])
        labels=y[tr];assert rec['training_labels']==labels.tolist()
        expected_splits=list(StratifiedKFold(3,shuffle=True,random_state=20261011).split(tr,labels))
        scores={}
        for c in CANDIDATES:
            train_x=features[c['feature']][tr];pred=np.full(len(tr),np.nan)
            for i,((it,iv),part) in enumerate(zip(expected_splits,rec['inner_partitions']),1):
                assert part['train_indices']==it.tolist() and part['validation_indices']==iv.tolist()
                assert part['training_subjects']==[subjects[tr[j]] for j in it]
                assert part['validation_subjects']==[subjects[tr[j]] for j in iv]
                model=joblib.load(output/f'fold_{fold}'/c['id']/f'inner_{i}.joblib');verified+=1
                check_model(model,OLD_MODELS[c['original_pipeline']],train_x[it])
                pred[iv]=model.predict_proba(train_x[iv])[:,1]
            saved=rec['candidates'][c['id']]
            np.testing.assert_array_equal(pred,saved['inner_probabilities'])
            scores[c['id']]=float(average_precision_score(labels,pred))
            assert scores[c['id']]==saved['inner_ap'] and threshold_rule(labels,pred,.8)==saved['rule']
            model=joblib.load(output/f'fold_{fold}'/c['id']/'outer.joblib');verified+=1
            check_model(model,OLD_MODELS[c['original_pipeline']],train_x)
            np.testing.assert_array_equal(model.predict_proba(features[c['feature']][va])[:,1],probabilities[c['id']][va])
        assert select(scores)==rec['selected']
    results=registry(output)
    table=pd.read_csv(output/'oof_predictions_private.csv',dtype={'subject_id':str})
    assert table.subject_id.tolist()==subjects
    np.testing.assert_array_equal(table.label,y);np.testing.assert_array_equal(table.fold,folds)
    results['exp4_joint_primary']=(table.exp4_joint_primary_probability.to_numpy(),table.exp4_joint_primary_threshold.to_numpy())
    metrics=json.loads((output/'metrics.json').read_text())
    for name,(p,t) in results.items():
        np.testing.assert_allclose(p,table[name+'_probability'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(t,table[name+'_threshold'],rtol=0,atol=1e-14)
        check_metrics(y,p,t,metrics['models'][name])
        for fold in range(1,6):check_metrics(y[folds==fold],p[folds==fold],t[folds==fold],metrics['models'][name]['per_fold'][str(fold)])
    # Ranking and class metrics are independently checked above. Recompute
    # uncertainty from the exact archived probability arrays, avoiding CSV rounding.
    old=pd.read_csv(root/'data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/oof_predictions.csv',dtype={'subject_id':str}).set_index('subject_id').loc[subjects]
    results['exp4_joint_primary']=(old.exp04_joint_selected_probability.to_numpy(),old.exp04_joint_selected_threshold.to_numpy())
    if compare(y,folds,results)!=metrics:raise ValueError('Conditional intervals/paired metrics differ')
    if stability(output)!=json.loads((output/'coefficient_stability.json').read_text()):raise ValueError('Stability summary differs')
    if fresh_features:
        rows={r['subject_id']:r for r in json.loads((output/'audit/fresh_inputs.json').read_text())}
        for i,sid in enumerate(subjects):
            row=rows[sid];rgb,anatomy=load_manual_pair(root/row['annotation_image_path'],root/row['mask_path'])
            glare,q=propose_glare(rgb,anatomy)
            with Image.open(output/'glare_masks'/f'E4B-{i+1:03d}.png') as image:np.testing.assert_array_equal(glare,np.asarray(image))
            for prefix,values in [('original',extract(rgb,anatomy)),('proposed_excluded',extract(rgb,anatomy,glare))]:
                for key,v in values.items():np.testing.assert_array_equal(v,all_features[prefix+'_'+key][i])
            if (i+1)%20==0:print(f'Freshly reproduced {i+1}/119 masks/features',flush=True)
    return {'status':'reproduced_without_refitting','classifier_pipelines':verified,'covered_files':len(expected),
            'inner_decisions_metrics_intervals_and_stability_verified':True,'fresh_masks_and_features_exact':fresh_features,
            'sources_previous_experiments_unchanged':True,'glare_exclusion_models_run':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fresh-features',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):print(json.dumps(reproduce(args.output,args.source_root,args.fresh_features),indent=2))
