"""Private cohort audit, glare proposals and controlled feature diagnostics."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from .eyelid_color_run import PRIOR_NAMES, write_json, prior_unchanged
from .eyelid_glare_masks import DETECTOR, propose_glare, panel
from .eyelid_glare_features import FEATURES, REGIONS, extract
from .local_eyelid_hybrid_features import validated_inputs
from .local_eyelid_validation import sha256, strict_csv, load_manual_pair, source_snapshot
from .preserve_local_experiment import tree_checksums

EXP4 = 'local_eyelid_classification_exp04_color_optimization_2026-10-12_v1'
PRESERVED = (*PRIOR_NAMES, EXP4, 'exp04_preflight_2026-10-11_v1', 'exp04_verification_2026-10-12_v1')


def destination(root: Path, output: Path) -> Path:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_exp04b_glare_regional_'):
        raise ValueError('Require distinct private Experiment 4B output')
    if output.exists(): raise FileExistsError('Never overwrite an experiment attempt')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')],cwd=root).returncode:
        raise ValueError('Private output is not ignored')
    return output


def coverage_audit(root: Path, subjects: list[str]) -> tuple[dict, list[dict]]:
    rows = [r for r in strict_csv(root/'data/raw/local/annotations/manifests/annotation_manifest.csv') if r['modality']=='eyelid']
    records = []
    for row in rows:
        path = row['glare_candidate_or_mask_path']; available = bool(path and (root/path).is_file())
        item = {'subject_id': row['subject_id'], 'eligible': row['subject_id'] in subjects,
                'existing_status': row['glare_status'], 'candidate_path': path,
                'candidate_exists': available, 'human_glare_approval': False,
                'missing_means': 'not_reviewed_or_missing_annotation; never no_glare'}
        if available:
            with Image.open(root/path) as image: glare = np.asarray(image)
            with Image.open(root/row['mask_path']) as image: anatomy = np.asarray(image)
            valid = glare.dtype==np.uint8 and glare.ndim==2 and glare.shape==anatomy.shape and np.isin(glare,[0,255]).all()
            item.update(candidate_sha256=sha256(root/path), binary_alignment_valid=bool(valid))
            if valid:
                item.update(marked_pixels=int((glare>0).sum()), outside_roi_pixels=int(((glare>0)&(anatomy==0)).sum()),
                            roi_removed_fraction=float(((glare>0)&(anatomy>0)).sum()/(anatomy>0).sum()))
        records.append(item)
    eligible = [r for r in records if r['eligible']]
    summary = {'approved_anatomical_eyelids': sum(r['annotation_status']=='approved' for r in rows),
               'eligible_eyelids': len(subjects), 'approved_glare_masks': 0, 'reviewed_no_glare': 0,
               'existing_candidates_all': sum(r['candidate_exists'] for r in records),
               'without_existing_mask_all': sum(not r['candidate_exists'] for r in records),
               'existing_candidates_eligible': sum(r['candidate_exists'] for r in eligible),
               'without_existing_mask_eligible': sum(not r['candidate_exists'] for r in eligible),
               'existing_definition_statuses': dict(Counter(r['glare_status'] for r in rows)),
               'valid_existing_candidates': sum(r.get('binary_alignment_valid',False) for r in records),
               'existing_candidates_outside_roi': sum(r.get('outside_roi_pixels',0)>0 for r in records),
               'definition': 'Batch1 unvalidated pilot rule; Batch2 unreviewed candidates. No accepted human glare review register or verified no-glare records.',
               'pale_tissue_assessment': 'Cannot establish tissue-vs-reflection accuracy or exclusion suitability without reference annotations. Existing masks are not ground truth.'}
    return summary, records


def validate_fresh_inputs(fresh: dict, data: dict) -> None:
    """Compare typed audit labels to CSV strings without weakening hash checks."""
    if set(fresh)!=set(data['subjects']): raise ValueError('Fresh cohort membership differs')
    for sid in data['subjects']:
        for key in ('original_source_sha256','annotation_image_sha256','roi_sha256','mask_sha256','derived_anemia_label'):
            if str(fresh[sid][key])!=str(data['rows'][sid][key]):
                raise ValueError('Fresh input provenance differs')


def prepare(root: Path, output: Path, preflight: Path) -> dict:
    cv2.setNumThreads(1); os.umask(0o077)
    root = root.resolve(); output = destination(root, output)
    canonical = root/'data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv'
    snapshot = source_snapshot(canonical,canonical.parent/'provenance.json')
    audit = json.loads((preflight/'fresh_audit.json').read_text())
    if audit['metadata_source']['sha256'] != snapshot['sha256']: raise ValueError('Fresh audit metadata mismatch')
    data = validated_inputs(root, root/'data/processed'/PRIOR_NAMES[0], root/'data/processed'/PRIOR_NAMES[1])
    fresh = {r['subject_id']:r for r in json.loads((preflight/'fresh_inputs.json').read_text())}
    validate_fresh_inputs(fresh,data)
    previous = {name:tree_checksums(root/'data/processed'/name) for name in PRESERVED}
    historical = pd.read_csv(root/'data/processed'/EXP4/'oof_predictions.csv',dtype={'subject_id':str}).set_index('subject_id').loc[data['subjects']]
    np.testing.assert_array_equal(historical.label,data['labels']); np.testing.assert_array_equal(historical.fold,data['folds'])
    source_hashes = json.loads((root/'data/processed'/EXP4/'experiment_config.json').read_text())['source_hashes']
    source_hashes.update(audit['source_files'])
    output.mkdir(mode=0o700)
    for name in ('glare_masks','review_panels','audit','implementation'): (output/name).mkdir(mode=0o700)
    summary, old_records = coverage_audit(root,data['subjects'])
    old_by_id = {r['subject_id']:r for r in old_records}
    write_json(output/'audit/existing_glare_audit.json',summary)
    pd.DataFrame(old_records).to_csv(output/'audit/existing_glare_records_private.csv',index=False)
    for r in old_records:
        if r['candidate_exists']: source_hashes[str(root/r['candidate_path'])] = r['candidate_sha256']
    collected = {f'{p}_{f}':[] for p in ('original','proposed_excluded') for f in FEATURES}
    records=[]; identity=[]
    for i,sid in enumerate(data['subjects']):
        row = fresh[sid]; rgb,anatomy = load_manual_pair(root/row['annotation_image_path'],root/row['mask_path'])
        glare,quality = propose_glare(rgb,anatomy)
        review_id = f'E4B-{i+1:03d}'
        mask_file = output/'glare_masks'/f'{review_id}.png'; Image.fromarray(glare).save(mask_file)
        original,excluded = extract(rgb,anatomy),extract(rgb,anatomy,glare)
        for key in FEATURES:
            collected[f'original_{key}'].append(original[key]); collected[f'proposed_excluded_{key}'].append(excluded[key])
        core = original['regional'][18:]
        yy,xx = np.where(anatomy>0)
        record={'review_id':review_id, **quality, 'glare_mask_sha256':sha256(mask_file),
                'source_rendering':'A_unchanged_stored_TIFF_RGB_assumed_sRGB',
                'existing_candidate':old_by_id[sid]['candidate_exists'],
                'roi_bbox_fill':float((anatomy>0).sum()/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))),
                **{name:float(value) for name,value in zip(FEATURES['regional'][18:],core)}}
        old = old_by_id[sid]
        if old.get('binary_alignment_valid'):
            with Image.open(root/old['candidate_path']) as image: oldmask=(np.asarray(image)>0)&(anatomy>0)
            union = (oldmask|(glare>0)).sum()
            record['iou_with_unapproved_candidate'] = float((oldmask&(glare>0)).sum()/union) if union else 1.
        panel(rgb,anatomy,glare,record,output/'review_panels'/f'{review_id}.png')
        records.append(record); identity.append({'review_id':review_id,'subject_id':sid})
        if (i+1)%20==0: print(f'Prepared {i+1}/119 glare proposals and review panels',flush=True)
    features={k:np.asarray(v,dtype=float) for k,v in collected.items()}
    with np.load(root/'data/processed'/EXP4/'features.npz',allow_pickle=False) as old:
        np.testing.assert_array_equal(old['subjects'],data['subjects'])
        np.testing.assert_array_equal(features['original_robust'],old['A_robust'])
    np.savez_compressed(output/'features.npz',subjects=np.asarray(data['subjects'],dtype=str),labels=data['labels'],folds=data['folds'],**features)
    frame=pd.DataFrame(records); frame.to_csv(output/'glare_coverage_private.csv',index=False)
    pd.DataFrame(identity).to_csv(output/'identity_key_private.csv',index=False)
    queue=frame.sort_values(['review_group','removed_fraction'],ascending=[True,False]).copy()
    queue['owner_decision']=''; queue['reviewer']=''; queue['review_date']=''; queue['reviewed_mask_sha256']=''; queue['notes']=''
    queue.to_csv(output/'review_queue_private.csv',index=False)
    summary.update(generated_masks=len(records),nonempty_generated_masks=int((frame.glare_pixels>0).sum()),
                   empty_pending_masks=int((frame.glare_pixels==0).sum()),
                   generated_status_counts=frame.status.value_counts().to_dict(),
                   review_group_counts=frame.review_group.value_counts().to_dict(),
                   masks_requiring_human_review=len(records),confirmed_glare_free=0,
                   invalid_generated_masks=0,outside_roi_pixels=int(frame.outside_roi_pixels.sum()),
                   removed_fraction={'median':float(frame.removed_fraction.median()),'max':float(frame.removed_fraction.max()),'total_pixel_fraction':float(frame.glare_pixels.sum()/frame.roi_pixels.sum())},
                   strict_verified_subset={'subjects':0,'anemia':0,'non_anemia':0,'state':'BLOCKED_DATA_NO_VERIFIED_GLARE_REVIEWS'},
                   full_cohort_exclusion={'state':'BLOCKED_PREPROCESSING_UNVALIDATED_GLARE_PROPOSALS','use':'diagnostics_only; no exclusion classifier fitting'},
                   anatomical_regional_models={'state':'READY_FOR_EXPLORATORY_TRAINING','subjects':119,'anemia':19,'non_anemia':100})
    write_json(output/'glare_audit_summary.json',summary)
    diagnostic(root,output,features,frame,data,historical,preflight)
    for path,digest in source_hashes.items():
        if sha256(Path(path))!=digest: raise ValueError('Source changed during proposal generation')
    prior_unchanged(root,previous)
    config={'canonical_snapshot_before_read':snapshot,'detector':DETECTOR,'regions':REGIONS,'feature_names':{k:list(v) for k,v in FEATURES.items()},
            'preprocessing':'Unchanged Exp4 A native TIFF RGB assumed sRGB; same manual mask; OpenCV uint8 CIELAB; no resize/enhancement/normalization. Proposed exclusions diagnostic only.',
            'source_hashes':source_hashes,'previous_checksums':previous,'exp4_commit':'0b5050436862df240fb60e1e93113e5c2388de98',
            'exp4_reproduction':{'models':720,'regression_models':10,'fresh_features_exact':True},
            'scope':'Exploratory repeated development subjects; no external validation; nonpregnancy owner-attested outside Sheet.'}
    write_json(output/'preparation_config.json',config)
    for path in preflight.glob('*.json'): shutil.copy2(path,output/'audit'/path.name)
    print(json.dumps(summary,indent=2)); return summary


def diagnostic(root: Path, output: Path, features: dict, frame: pd.DataFrame, data: dict, historical: pd.DataFrame, preflight: Path) -> None:
    clinical={r['subject_id']:r for r in json.loads((preflight/'fresh_eligibility.json').read_text())}
    p=historical.ablation_A_robust_lr_c1_none_probability.to_numpy();t=historical.ablation_A_robust_lr_c1_none_threshold.to_numpy()
    y=data['labels']; groups=np.where(y==1,np.where(p>=t,'TP','FN'),np.where(p>=t,'FP','TN'))
    result=frame.copy();result['group']=groups;result['label']=y;result['fold']=data['folds'];result['exp4_score']=p;result['exp4_threshold']=t
    result['exp01_fixed_detected']=historical.exp01_fixed_probability.to_numpy()>=.5
    result['historical_lab_selected_detected']=historical.lab_nested_selected_probability.to_numpy()>=historical.lab_nested_selected_threshold.to_numpy()
    result['hb_g_dl']=[float(clinical[s]['normalized_hb_g_per_dl']) for s in data['subjects']]
    result['cutoff_minus_hb']=[(13 if clinical[s]['sex']=='M' else 12)-float(clinical[s]['normalized_hb_g_per_dl']) for s in data['subjects']]
    result['severity']=np.where(y==0,'non_anemia',np.where(result.hb_g_dl>=11,'mild',np.where(result.hb_g_dl>=8,'moderate','severe')))
    result['render_history']=[data['rows'][s]['render_method_recorded'] for s in data['subjects']]
    for j,name in enumerate(FEATURES['robust']): result[name]=features['original_robust'][:,j]
    result.to_csv(output/'diagnostics_private.csv',index=False)
    difference=features['proposed_excluded_robust']-features['original_robust']
    perturbation=[]
    for j,name in enumerate(FEATURES['robust']):
        sd=float(features['original_robust'][:,j].std())
        perturbation.append({'feature':name,'median_absolute_delta':float(np.median(np.abs(difference[:,j]))),
                             'max_absolute_delta':float(np.max(np.abs(difference[:,j]))),
                             'median_delta_in_cohort_sd':float(np.median(np.abs(difference[:,j]))/sd) if sd else None,
                             'original_between_subject_std':sd,'proposed_excluded_between_subject_std':float(features['proposed_excluded_robust'][:,j].std())})
    summaries={}
    for name in ('FN','TP','non_anemia'):
        sub=result[result.group.isin(['FP','TN'])] if name=='non_anemia' else result[result.group==name]
        numbers=['hb_g_dl','cutoff_minus_hb','removed_fraction','roi_pixels','roi_bbox_fill','median_L','median_a','median_log_red_green',*FEATURES['regional'][18:]]
        summaries[name]={'n':len(sub),'severity':sub.severity.value_counts().to_dict(),'render_history':sub.render_history.value_counts().to_dict(),
                         'nonempty_proposals':int((sub.glare_pixels>0).sum()),'uncertain_proposals':int((sub.status=='UNCERTAIN').sum()),
                         'medians':{k:float(sub[k].median()) for k in numbers}}
    assert summaries['FN']['n']==9 and summaries['TP']['n']==10
    summary={'groups':summaries,'proposed_exclusion_perturbation_diagnostics_only':perturbation,
             'complementary_errors':{'robust_recovers_historical_lab_misses':int(((y==1)&(p>=t)&~result.historical_lab_selected_detected).sum()),
                                    'robust_loses_historical_lab_detections':int(((y==1)&(p<t)&result.historical_lab_selected_detected).sum()),
                                    'robust_recovers_exp01_fixed_misses':int(((y==1)&(p>=t)&~result.exp01_fixed_detected).sum()),
                                    'robust_loses_exp01_fixed_detections':int(((y==1)&(p<t)&result.exp01_fixed_detected).sum())},
             'glare_association':'Unvalidated proposal burden is not actual glare. Descriptive error-conditioned comparisons cannot establish glare as the cause of missed anemia. No diagnostic was used to tune detector/features.'}
    write_json(output/'diagnostic_summary.json',summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args();prepare(args.source_root,args.output,args.preflight)
