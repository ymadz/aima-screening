"""Owner-reviewed, controlled original versus glare-excluded Experiment4B."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .eyelid_color_run import write_json,prior_unchanged
from .eyelid_color_models import threshold_rule
from .eyelid_color_verify import check_model,check_metrics
from .eyelid_color_models import BY_ID as OLD_MODELS
from .eyelid_glare_features import FEATURES,require_verified_review,extract
from .eyelid_glare_masks import validate_glare
from .eyelid_glare_models import fit_outer,CANDIDATES
from .eyelid_glare_evaluation import compare
from .eyelid_glare_run import check_exp4_comparator
from .local_eyelid_validation import sha256,strict_csv,load_manual_pair
from .preserve_local_experiment import tree_checksums
from PIL import Image

GROUPS=('original','glare_excluded')
ORDER=[f'{g}_{c["id"]}' for g in GROUPS for c in CANDIDATES]
DESIGN={'groups':list(GROUPS),'candidate_order':ORDER,'inner_seed':20261011,'inner_folds':3,
        'outer_folds':'unchanged119-subject original five folds','models':'Exact previous4B robust18/regional22 C1 logistic none/balanced pipelines',
        'selection':'highest inner pooled AP; frozen group/candidate order ties',
        'threshold':'inner sensitivity maximum subject to specificity>=0.80, then specificity and cutoff',
        'fixed_threshold':.5,'mask_policy':'Only owner-accepted exact native masks; empty accepted masks REVIEWED_NO_GLARE; retain original anatomy and original regions',
        'scope':'Strict reviewed subset equals full eligible cohort119 (19/100); same analysis, not independent evidence. Owner visual QC, not clinician/reference-mask accuracy validation.'}


def verify_tree(path: Path) -> None:
    expected=json.loads((path/'checksum_manifest.json').read_text());actual=tree_checksums(path);actual.pop('checksum_manifest.json',None)
    if expected!=actual:raise ValueError('Checksum coverage/content differs; no deserialization')


def selected(scores: dict, group: str | None=None) -> str:
    eligible=[k for k in ORDER if group is None or k.startswith(group+'_')]
    if set(scores)!=set(ORDER):raise ValueError('Incomplete candidate scores')
    return max(eligible,key=lambda k:(scores[k],-ORDER.index(k)))


def inputs(source: Path, reviewed: Path) -> tuple[dict,dict,list[dict]]:
    verify_tree(source);verify_tree(reviewed)
    rows=strict_csv(reviewed/'human_glare_review_register_private.csv');require_verified_review(rows)
    with np.load(source/'features.npz',allow_pickle=False) as cache:
        metadata={k:cache[k] for k in ('subjects','labels','folds')}
        features={g:{f:cache[('original' if g=='original' else 'proposed_excluded')+'_'+f] for f in FEATURES} for g in GROUPS}
    if len(metadata['subjects'])!=119 or int(metadata['labels'].sum())!=19 or set(metadata['folds'])!={1,2,3,4,5}:
        raise ValueError('Original cohort/class/fold support differs')
    by_id={r['subject_id']:r for r in rows}
    if len(rows)!=119 or set(by_id)!=set(metadata['subjects']):raise ValueError('Review must cover exact cohort without duplicates')
    for sid in metadata['subjects']:
        r=by_id[sid];path=Path(r['reviewed_mask_path'])
        if sha256(path)!=r['reviewed_mask_sha256'] or sha256(Path(r['review_package_path']))!=r['review_package_sha256']:
            raise ValueError('Reviewed bytes differ from accepted package')
        with Image.open(path) as im:mask=np.asarray(im)
        expected_status='APPROVED_GLARE' if np.any(mask) else 'REVIEWED_NO_GLARE'
        if r['status']!=expected_status:raise ValueError('Reviewed glare/empty status differs')
    return metadata,features,rows


def score_registry(output: Path) -> dict:
    with np.load(output/'features_used.npz',allow_pickle=False) as cache:folds=cache['folds']
    with np.load(output/'candidate_predictions.npz',allow_pickle=False) as cache:prob={k:cache[k] for k in cache.files}
    result={}
    for name in ORDER:
        cutoff=np.full(len(folds),np.nan)
        for fold in range(1,6):
            rec=json.loads((output/f'fold_{fold}/selection.json').read_text())
            group,candidate=name.split('_',1) if name.startswith('original_') else ('glare_excluded',name.removeprefix('glare_excluded_'))
            cutoff[folds==fold]=rec['groups'][group]['candidates'][candidate]['rule']['threshold']
        result[name+'_fixed']=(prob[name],np.full(len(folds),.5));result[name+'_selected']=(prob[name],cutoff)
    for policy in ('original_selected','glare_excluded_selected','full_selected'):
        p=np.full(len(folds),np.nan);t=p.copy()
        for fold in range(1,6):
            rec=json.loads((output/f'fold_{fold}/selection.json').read_text());name=rec['policies'][policy]
            group,candidate=name.split('_',1) if name.startswith('original_') else ('glare_excluded',name.removeprefix('glare_excluded_'))
            p[folds==fold]=prob[name][folds==fold];t[folds==fold]=rec['groups'][group]['candidates'][candidate]['rule']['threshold']
        result[policy]=(p,t);result[policy+'_fixed']=(p,np.full(len(folds),.5))
    result['robust_none_selected']=result['original_robust_none_selected']
    return result


def reviewed_stability(output: Path) -> dict:
    result={}
    for group in GROUPS:
        for c in CANDIDATES:
            values=[]
            for fold in range(1,6):
                model=joblib.load(output/f'fold_{fold}'/group/c['id']/'outer.joblib');values.append(model['classifier'].coef_[0])
            values=np.asarray(values)
            result[f'{group}_{c["id"]}']=[{'feature':f,'coefficient_by_fold':values[:,j].tolist(),
                'dominant_sign_fraction':float(max((values[:,j]>0).sum(),(values[:,j]<0).sum())/5)} for j,f in enumerate(FEATURES[c['feature']])]
    return result


def run(root: Path, source: Path, reviewed: Path, output: Path) -> dict:
    cv2.setNumThreads(1);os.umask(0o077);root=root.resolve();output=output.resolve();start=time.monotonic()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_exp04b_reviewed_comparison_') or output.exists():
        raise ValueError('Require new versioned private reviewed comparison output')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')],cwd=root).returncode:raise ValueError('Output not ignored')
    metadata,features,reviews=inputs(source,reviewed)
    preparation=json.loads((source/'preparation_config.json').read_text())
    prior_unchanged(root,preparation['previous_checksums'])
    for path,digest in preparation['source_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError('Sources changed after verified preparation')
    # Revalidate every accepted mask against source anatomy and freshly extracted
    # original/excluded features; no clinical outcome influences mask acceptance.
    rows={r['subject_id']:r for r in json.loads((source/'audit/fresh_inputs.json').read_text())}
    for i,sid in enumerate(metadata['subjects']):
        row=rows[sid];rgb,anatomy=load_manual_pair(root/row['annotation_image_path'],root/row['mask_path'])
        review=next(r for r in reviews if r['subject_id']==sid)
        with Image.open(review['reviewed_mask_path']) as image:glare=np.asarray(image)
        validate_glare(rgb,anatomy,glare)
        for group,pixels in [('original',None),('glare_excluded',glare)]:
            for name,x in extract(rgb,anatomy,pixels).items():np.testing.assert_array_equal(x,features[group][name][i])
        if (i+1)%20==0:print(f'Fresh owner-reviewed mask/features validated {i+1}/119',flush=True)
    output.mkdir(mode=0o700);(output/'implementation/src').mkdir(parents=True,mode=0o700)
    repo=Path(__file__).resolve().parents[1];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    code={str(p.relative_to(repo)):sha256(p) for p in (repo/'src').glob('*.py')}
    for rel in code:shutil.copy2(repo/rel,output/'implementation'/rel)
    np.savez_compressed(output/'features_used.npz',**metadata,**{f'{g}_{f}':v for g,d in features.items() for f,v in d.items()})
    write_json(output/'experiment_config.json',{'design':DESIGN,'training_commit':commit,'code_sha256':code,
              'source_run':str(source.resolve()),'source_run_checksums':tree_checksums(source),
              'review_bundle':str(reviewed.resolve()),'review_bundle_checksums':tree_checksums(reviewed),
              'approved_glare':96,'reviewed_no_glare':23,'strict_reviewed_subset_equals_full_cohort':True,
              'source_hashes':preparation['source_hashes'],'previous_checksums':preparation['previous_checksums']})
    shutil.copy2(reviewed/'human_glare_review_register_private.csv',output/'human_glare_review_register_private.csv')
    y,folds,subjects=metadata['labels'],metadata['folds'],metadata['subjects'].tolist();prob={k:np.full(119,np.nan) for k in ORDER}
    for fold in range(1,6):
        tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold);directory=output/f'fold_{fold}';directory.mkdir(mode=0o700)
        records={};scores={}
        for group in GROUPS:
            target=directory/group;target.mkdir(mode=0o700)
            p,rec=fit_outer({k:v[tr] for k,v in features[group].items()},y[tr],[subjects[i] for i in tr],{k:v[va] for k,v in features[group].items()},target)
            records[group]=rec
            for c in CANDIDATES:
                name=f'{group}_{c["id"]}';prob[name][va]=p[c['id']];scores[name]=rec['candidates'][c['id']]['inner_ap']
        policies={f'{g}_selected':selected(scores,g) for g in GROUPS};policies['full_selected']=selected(scores)
        write_json(directory/'selection.json',{'train_indices':tr.tolist(),'validation_indices':va.tolist(),'groups':records,'policies':policies})
        print(f'Outer fold{fold}: {policies}',flush=True)
    np.savez_compressed(output/'candidate_predictions.npz',**prob);registry=score_registry(output)
    check_exp4_comparator(root/'data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1',folds,registry['robust_none_selected'])
    metrics=compare(y,folds,registry);write_json(output/'metrics.json',metrics);write_json(output/'coefficient_stability.json',reviewed_stability(output))
    table=pd.DataFrame({'subject_id':subjects,'label':y,'fold':folds})
    for name,(p,t) in registry.items():table[name+'_probability']=p;table[name+'_threshold']=t
    table.to_csv(output/'oof_predictions_private.csv',index=False)
    prior_unchanged(root,preparation['previous_checksums']);verify_tree(source);verify_tree(reviewed)
    write_json(output/'run_manifest.json',{'status':'STRICT_REVIEWED_AND_FULL_COHORT_COMPLETE','subjects':119,'anemia':19,'non_anemia':100,'models':160,'approved_glare':96,'reviewed_no_glare':23,'elapsed_seconds_fresh_feature_validation_training_evaluation':time.monotonic()-start,'clinical_validation':False})
    (output/'REPRODUCTION.md').write_text(f'From the Experiment4B worktree, project Python3.10 and one CPU thread:\n\npython -m src.eyelid_glare_reviewed --reproduce --source-root {root} --output {output}\n\nChecksum-first reproduction of160 saved models, inner decisions, exact folds and metrics/intervals; no refitting. Source features were freshly checked for all119 before training and the prior sealed anatomy run retains independent fresh reproduction. Owner review is visual dataset QC, not a clinical accuracy benchmark.\n')
    for path in output.rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
    write_json(output/'checksum_manifest.json',tree_checksums(output))
    return {'models':160,'subjects':119,'status':'sealed'}


def reproduce(root: Path, output: Path) -> dict:
    verify_tree(output);config=json.loads((output/'experiment_config.json').read_text())
    if config['design']!=DESIGN:raise ValueError('Frozen design differs')
    for key,pathkey in [('source_run_checksums','source_run'),('review_bundle_checksums','review_bundle')]:
        if tree_checksums(Path(config[pathkey]))!=config[key]:raise ValueError('Source/review artifact changed')
    for path,digest in config['source_hashes'].items():
        if sha256(Path(path))!=digest:raise ValueError('Source changed')
    prior_unchanged(root,config['previous_checksums'])
    repo=Path(__file__).resolve().parents[1]
    for path,digest in config['code_sha256'].items():
        if sha256(repo/path)!=digest or sha256(output/'implementation'/path)!=digest:raise ValueError('Implementation differs')
        import hashlib
        if hashlib.sha256(subprocess.check_output(['git','show',f"{config['training_commit']}:{path}"],cwd=root)).hexdigest()!=digest:raise ValueError('Recorded commit differs')
    with np.load(output/'features_used.npz',allow_pickle=False) as cache:
        y,folds,subjects=cache['labels'],cache['folds'],cache['subjects'].tolist()
        features={g:{f:cache[f'{g}_{f}'] for f in FEATURES} for g in GROUPS}
    metadata,source_features,reviews=inputs(Path(config['source_run']),Path(config['review_bundle']))
    for group in GROUPS:
        for name,x in features[group].items():np.testing.assert_array_equal(x,source_features[group][name])
    for name,x in [('labels',y),('folds',folds),('subjects',subjects)]:np.testing.assert_array_equal(x,metadata[name])
    with np.load(output/'candidate_predictions.npz',allow_pickle=False) as cache:prob={k:cache[k] for k in cache.files}
    count=0
    for fold in range(1,6):
        rec=json.loads((output/f'fold_{fold}/selection.json').read_text());tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        assert rec['train_indices']==tr.tolist() and rec['validation_indices']==va.tolist();scores={}
        for group in GROUPS:
            r=rec['groups'][group];assert r['training_subjects']==[subjects[i] for i in tr] and r['training_labels']==y[tr].tolist()
            from sklearn.model_selection import StratifiedKFold
            splits=list(StratifiedKFold(3,shuffle=True,random_state=20261011).split(tr,y[tr]))
            for c in CANDIDATES:
                x=features[group][c['feature']][tr];inner=np.full(len(tr),np.nan)
                for i,((it,iv),part) in enumerate(zip(splits,r['inner_partitions']),1):
                    assert part['train_indices']==it.tolist() and part['validation_indices']==iv.tolist()
                    assert part['training_subjects']==[subjects[tr[j]] for j in it]
                    assert part['validation_subjects']==[subjects[tr[j]] for j in iv]
                    assert set(part['training_subjects']).isdisjoint(part['validation_subjects'])
                    model=joblib.load(output/f'fold_{fold}'/group/c['id']/f'inner_{i}.joblib');count+=1
                    check_model(model,OLD_MODELS[c['original_pipeline']],x[it]);inner[iv]=model.predict_proba(x[iv])[:,1]
                detail=r['candidates'][c['id']];np.testing.assert_array_equal(inner,detail['inner_probabilities'])
                scores[f'{group}_{c["id"]}']=float(average_precision_score(y[tr],inner))
                assert scores[f'{group}_{c["id"]}']==detail['inner_ap'] and threshold_rule(y[tr],inner,.8)==detail['rule']
                model=joblib.load(output/f'fold_{fold}'/group/c['id']/'outer.joblib');count+=1
                check_model(model,OLD_MODELS[c['original_pipeline']],x)
                np.testing.assert_array_equal(model.predict_proba(features[group][c['feature']][va])[:,1],prob[f'{group}_{c["id"]}'][va])
        assert rec['policies']=={**{f'{g}_selected':selected(scores,g) for g in GROUPS},'full_selected':selected(scores)}
    registry=score_registry(output);saved=json.loads((output/'metrics.json').read_text())
    table=pd.read_csv(output/'oof_predictions_private.csv',dtype={'subject_id':str});assert table.subject_id.tolist()==subjects
    for name,(p,t) in registry.items():
        np.testing.assert_allclose(table[name+'_probability'],p,rtol=0,atol=1e-14);np.testing.assert_allclose(table[name+'_threshold'],t,rtol=0,atol=1e-14)
        check_metrics(y,p,t,saved['models'][name])
        for fold in range(1,6):check_metrics(y[folds==fold],p[folds==fold],t[folds==fold],saved['models'][name]['per_fold'][str(fold)])
    if compare(y,folds,registry)!=saved:raise ValueError('Metrics/intervals differ')
    if reviewed_stability(output)!=json.loads((output/'coefficient_stability.json').read_text()):raise ValueError('Stability differs')
    return {'status':'reproduced_without_refitting','models':count,'covered_files':len(json.loads((output/'checksum_manifest.json').read_text())),'reviewed_subjects':119,'inner_decisions_metrics_intervals_and_stability_verified':True,'all_source_features_fresh_verified_before_training':True,'prior_outputs_and_reviews_unchanged':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path);parser.add_argument('--reviewed',type=Path);parser.add_argument('--reproduce',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):
        result=reproduce(args.source_root.resolve(),args.output.resolve()) if args.reproduce else run(args.source_root,args.source,args.reviewed,args.output)
    print(json.dumps(result,indent=2))
