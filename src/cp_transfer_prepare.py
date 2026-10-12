"""Exclusive private Experiment5 preparation and domain-comparison evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw
from .cp_transfer_data import PREPROCESSING,audit_public,verify_official_v2,tissue,describe
from .local_eyelid_hybrid_features import validated_inputs
from .local_eyelid_validation import source_snapshot,sha256,perceptual_hash
from .eyelid_transfer_data import local_pixels
from .eyelid_color_data import color_features
from .eyelid_color_run import write_json,prior_unchanged
from .eyelid_glare_prepare import validate_fresh_inputs


def destination(root: Path,output: Path) -> Path:
    root,output=root.resolve(),output.resolve()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_classification_exp05_cp_transfer_'):
        raise ValueError('Require private versioned Experiment5 output')
    if output.exists():raise FileExistsError('Never overwrite an experiment attempt')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')],cwd=root).returncode:raise ValueError('Output must be ignored')
    return output


def prepare(root: Path,public: Path,archive: Path,archive_hash: str,preflight: Path,output: Path) -> dict:
    os.umask(0o077);cv2.setNumThreads(1);root=root.resolve();output=destination(root,output)
    canonical=root/'data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv'
    snapshot=source_snapshot(canonical,canonical.parent/'provenance.json')
    previous=json.loads((preflight/'previous_checksums.json').read_text());prior_unchanged(root,previous)
    official=verify_official_v2(public,archive,archive_hash)
    records,audit=audit_public(public)
    data=validated_inputs(root,root/'data/processed/local_eyelid_classification_2026-10-10_v1',root/'data/processed/local_eyelid_readiness_2026-10-10_v2')
    fresh={r['subject_id']:r for r in json.loads((preflight/'fresh_inputs.json').read_text())};validate_fresh_inputs(fresh,data)
    prior_audit=json.loads((preflight/'fresh_audit.json').read_text())
    if prior_audit['metadata_source']['sha256']!=snapshot['sha256']:raise ValueError('Fresh source audit mismatch')
    clinical={r['subject_id']:r for r in json.loads((preflight/'fresh_eligibility.json').read_text())}
    public_x=[];public_stats=[];local_x=[];local_stats=[];local_lab=[];tensors_hashes={};thumbnails=[]
    for i,r in enumerate(records):
        with Image.open(r['path']) as image:values=np.asarray(image).copy()
        x,mask=tissue(values[:,:,:3],values[:,:,3]);public_x.append(x);public_stats.append({'image_id':r['image_id'],**describe(values[:,:,:3],mask)})
        tensors_hashes['cp/'+r['image_id']]=sha256_tensor(x)
        if i<8:thumbnails.append(('Public '+str(i+1),x))
    source_hashes={**audit['source_hashes'],str(archive.resolve()):archive_hash,**prior_audit['source_files']}
    for i,sid in enumerate(data['subjects']):
        row=fresh[sid];rgb,mask,rendering=local_pixels(root,row);x,_=tissue(rgb,mask)
        local_x.append(x);local_stats.append({'review_id':f'E5-{i+1:03d}',**rendering,**describe(rgb,mask)})
        from .local_eyelid_validation import load_manual_pair
        raw,raw_mask=load_manual_pair(root/row['annotation_image_path'],root/row['mask_path'])
        local_lab.append(color_features(raw,raw_mask)['robust'])
        tensors_hashes['local/'+sid]=sha256_tensor(x)
        for key,hashkey in [('source_image_path','original_source_sha256'),('annotation_image_path','annotation_image_sha256'),('roi_path','roi_sha256'),('mask_path','mask_sha256')]:source_hashes[str(root/row[key])]=row[hashkey]
        if i<8:thumbnails.append((f'Local E5-{i+1:03d}',x))
        if (i+1)%20==0:print(f'Prepared local masked inputs {i+1}/119',flush=True)
    if len(set(tensors_hashes.values()))!=len(tensors_hashes):raise ValueError('Exact processed tensor duplicate within/across domains')
    with np.load(root/'data/processed/local_eyelid_classification_exp04_color_optimization_2026-10-12_v1/features.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(np.asarray(local_lab),saved['A_robust']);np.testing.assert_array_equal(data['subjects'],saved['subjects'])
    # Screen original public raster versus original local full photograph; this
    # is a candidate screen, never identity proof. Shared tissue geometry can
    # make cropped-input hashes misleading, so exact tensor hashes are separate.
    public_fp=[]
    for r in records:
        with Image.open(r['path']) as im:values=np.asarray(im).copy();rgb=values[:,:,:3];rgb[values[:,:,3]==0]=128
        public_fp.append(perceptual_hash(rgb))
    cross=[]
    from .local_eyelid_validation import load_manual_pair
    for sid in data['subjects']:
        row=fresh[sid];raw,_=load_manual_pair(root/row['annotation_image_path'],root/row['mask_path']);fp=perceptual_hash(raw)
        for record,other in zip(records,public_fp):
            distance=int(np.count_nonzero(fp!=other))
            if distance<=6:cross.append({'public_image':record['image_id'],'local_subject':sid,'distance':distance})
    if cross:raise ValueError('Cross-domain image-similarity candidates require review')
    output.mkdir(mode=0o700)
    for name in ['audit','domain_comparison','implementation']:(output/name).mkdir(mode=0o700)
    y=data['labels'];hb=np.asarray([float(clinical[s]['normalized_hb_g_per_dl']) for s in data['subjects']]);sex=np.asarray([clinical[s]['sex'] for s in data['subjects']]);ages=[int(clinical[s]['age']) for s in data['subjects']]
    cache={'public_x':torch.stack(public_x),'public_y':torch.tensor([r['label'] for r in records]),'public_hb':torch.tensor([r['hb_g_dl'] for r in records],dtype=torch.float32),
           'public_ids':[r['image_id'] for r in records],'public_groups':[r['split_group'] for r in records],
           'public_train':audit['training_indices'],'public_validation':audit['validation_indices'],
           'local_x':torch.stack(local_x),'local_y':torch.tensor(y),'local_hb':torch.tensor(hb,dtype=torch.float64),
           'local_sex':sex.tolist(),'local_subjects':data['subjects'],'local_folds':data['folds'].tolist(),'local_lab':torch.tensor(np.asarray(local_lab),dtype=torch.float64)}
    torch.save(cache,output/'inputs.pt');pd.DataFrame(records).to_csv(output/'audit/public_manifest.csv',index=False)
    write_json(output/'audit/public_audit.json',{**audit,'authenticity':official,'cross_domain_candidates':cross})
    domain={'public':aggregate(public_stats),'local':aggregate(local_stats),'public_alpha':{'median_partial_fraction':float(np.median([r['alpha_boundary_fraction'] for r in records])),'max_partial_fraction':float(max(r['alpha_boundary_fraction'] for r in records))},
           'local_labels':{'subjects':119,'anemia':int(y.sum()),'non_anemia':int((y==0).sum()),'age_range':[min(ages),max(ages)],'hb_min':float(hb.min()),'hb_max':float(hb.max()),'hb_median':float(np.median(hb)),
             'anemia_severity_descriptive':{'mild':int(((y==1)&(hb>=11)).sum()),'moderate':int(((y==1)&(hb>=8)&(hb<11)).sum()),'severe':int(((y==1)&(hb<8)).sum())},'label_policy':'Unchanged WHO2024 unadjusted15-65 M<13,F<12; nonpregnancy owner-attested outside Sheet'},
           'limits':'Public RGB untagged/uncalibrated; alpha extraction algorithm differs from local human anatomy. Bright proxies are not actual glare annotations. Demographics unavailable per public record. Pixel scale/orientation/acquisition remain domain shifts even with shared processing.'}
    write_json(output/'domain_comparison/summary.json',domain)
    pd.DataFrame(public_stats).to_csv(output/'domain_comparison/public_statistics.csv',index=False);pd.DataFrame(local_stats).to_csv(output/'domain_comparison/local_statistics_private.csv',index=False)
    from .local_eyelid_hybrid_features import MEAN,STD
    sheet=Image.new('RGB',(320*4,200*4),'white')
    for i,(label,x) in enumerate(thumbnails):
        rgb=((x*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None])*255).clamp(0,255).byte().permute(1,2,0).numpy();image=Image.fromarray(rgb);image.thumbnail((300,160));sheet.paste(image,((i%4)*320+(320-image.width)//2,(i//4)*200+30));ImageDraw.Draw(sheet).text(((i%4)*320+8,(i//4)*200+8),label,fill='black')
    sheet.save(output/'domain_comparison/shared_inputs_private.png')
    (output/'domain_comparison/report.md').write_text('# CP-AnemiC/AIMA domain comparison\n\nPublic study pediatric Hb<11 and local adult/adolescent sex-specific labels stay separate. Public patient count is not independently established; group-held-image validation is not verified patient-level clinical testing. Source distribution and alpha/camera limitations persist after common tissue crop, mean background,224 geometry and fixed ImageNet normalization. See summary.json and private statistical tables for RGB/LAB, area/resolution, contrast/saturation/glare proxies. Partial alpha is excluded for opaque-core model input; it is not a new anatomical segmenter. No color normalization or glare exclusion is learned.\n')
    for path,digest in source_hashes.items():
        if sha256(Path(path))!=digest:raise ValueError('Source changed during preparation')
    prior_unchanged(root,previous)
    write_json(output/'preparation_config.json',{'preprocessing':PREPROCESSING,'canonical_snapshot_before_read':snapshot,'public_release':2,'public_root':str(public.resolve()),'source_root':str(root),
        'source_hashes':source_hashes,'previous_checksums':previous,'tensor_sha256':tensors_hashes,'inputs_sha256':sha256(output/'inputs.pt'),'subject_id_limit':audit['group_limit']})
    for p in preflight.glob('*.json'):shutil.copy2(p,output/'audit'/p.name)
    print(json.dumps({'public_images':len(records),'public_class_counts':audit['class_counts_0_1'],'public_split_groups':audit['split_groups'],'public_validation_class_counts':audit['validation_class_counts'],'local_subjects':119,'local_anemia':19,'official_local_files_match':official['local_files_exactly_matching_official']}))
    return {'public_images':len(records),'local_subjects':119,'status':'prepared'}


def sha256_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def aggregate(rows: list[dict]) -> dict:
    keys=['tissue_pixels','height','width','bbox_width','bbox_height','saturation_mean','brightness_mean','gray_contrast_std','bright_neutral_proxy_fraction','any_channel_clipped_fraction']
    result={k:{'median':float(np.median([r[k] for r in rows])),'min':float(min(r[k] for r in rows)),'max':float(max(r[k] for r in rows))} for k in keys}
    for k in ['rgb_mean','rgb_std','lab_mean','lab_std']:result[k+'_median']=np.median([r[k] for r in rows],axis=0).tolist()
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--public-root',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--archive-sha256',required=True);parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import hashlib
    prepare(args.source_root,args.public_root,args.archive,args.archive_sha256,args.preflight,args.output)
