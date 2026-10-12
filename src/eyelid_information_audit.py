"""Private descriptive eyelid audit; no fitting, ROI changes or clinical relabeling."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .local_eyelid_validation import sha256, strict_csv, load_manual_pair
from .eyelid_ensemble_run import write_json

SEVERITY_SOURCE = 'https://www.ncbi.nlm.nih.gov/books/NBK602185/table/ch5.tab3/'


def severity(hb: float, sex: str) -> str:
    """Unadjusted WHO2024 adult 15–65 description, never a replacement label."""
    if sex not in {'M', 'F'} or not np.isfinite(hb) or hb <= 0:
        raise ValueError('Unverified Hb or sex')
    if hb >= (13 if sex == 'M' else 12): return 'non_anemia'
    if hb >= 11: return 'mild'
    if hb >= 8: return 'moderate'
    return 'severe'


def distribution(values: pd.Series) -> dict:
    return {'median': float(values.median()), 'min': float(values.min()), 'max': float(values.max()),
            'q25':float(values.quantile(.25)), 'q75':float(values.quantile(.75))}


def audit(root: Path, output: Path) -> dict:
    """Use previously fresh-verified linkage and preserve every original/mask."""
    root,output=root.resolve(),output.resolve()
    if not output.is_relative_to(root/'data/processed') or not output.name.startswith('local_eyelid_classification_exp06_ensemble_v'):
        raise ValueError('Image audit requires the private Experiment6 bundle')
    if subprocess.run(['git','check-ignore','-q',str(output/'private.csv')],cwd=root).returncode:
        raise ValueError('Image audit output must be ignored')
    directory=output/'image_information';directory.mkdir(mode=0o700)
    inputs={r['subject_id']:r for r in json.loads((output/'verified_inputs.json').read_text())}
    clinical=pd.DataFrame(json.loads((output/'fresh_eligibility.json').read_text())).set_index('subject_id')
    keys=pd.read_csv(output/'identity_key_private.csv',dtype=str)
    oof=pd.read_csv(output/'oof_predictions_private.csv',float_precision='round_trip').set_index('deidentified_id')
    source_inventory=root/'data/raw/local/manifest/frozen_2026-10-07/local_image_manifest.csv'
    inventory=strict_csv(source_inventory)
    incoming=root/'data/raw/local/incoming'
    actual={p.name:p for p in incoming.iterdir() if p.is_file()}
    if len(actual)!=len(inventory) or set(actual)!={r['original_filename'] for r in inventory}:
        raise ValueError('Incoming inventory changed; alternate linkage cannot be inferred')
    for r in inventory:
        if sha256(actual[r['original_filename']])!=r['sha256']:
            raise ValueError('Original archive bytes differ from registered inventory')
    eyelids=[r for r in inventory if r['modality']=='eyelid']
    counts=Counter(r['subject_id'] for r in eyelids if r['subject_id'])
    repeated={s:n for s,n in counts.items() if n>1}
    repeats={'archive_files':len(inventory),'verified_eyelid_originals':len(eyelids),
             'subjects_with_multiple_linked_eyelid_originals':len(repeated),
             'unassigned_files_requiring_role_and_identity_evidence':sum(not r['subject_id'] for r in inventory),
             'inventory_path':str(source_inventory),'inventory_sha256':sha256(source_inventory),
             'all_archive_source_hashes_match':True,
             'repeated_prediction_consistency':'not estimable; one provenance-linked anatomical eyelid photograph per subject. Unassigned files cannot be assumed to be retakes; no alternate reviewed masks exist.',
             'fold_rule':'all source-linked images of a subject must retain its existing subject fold; never independent patients'}
    write_json(directory/'repeated_images.json',repeats)
    records=[];tiles=[]
    for _,key in keys.iterrows():
        sid,deid=key.source_subject_id,key.deidentified_id;row=inputs[sid];c=clinical.loc[sid];pred=oof.loc[deid]
        for field,path_field in [('original_source_sha256','source_image_path'),('annotation_image_sha256','annotation_image_path'),('roi_sha256','roi_path'),('mask_sha256','mask_path')]:
            if sha256(root/row[path_field])!=row[field]:raise ValueError('Approved source/ROI/mask bytes changed')
        image,mask=load_manual_pair(root/row['annotation_image_path'],root/row['mask_path']);sel=mask>0;yy,xx=np.where(sel)
        n,_,stats,_=cv2.connectedComponentsWithStats(sel.astype(np.uint8),connectivity=8)
        hb=float(c.normalized_hb_g_per_dl);cutoff=13 if c.sex=='M' else 12
        if int(pred.label)!=int(c.derived_anemia_label) or int(hb<cutoff)!=int(pred.label):raise ValueError('Clinical label mismatch')
        lab=int(pred.exp04_robust_historical_probability>=pred.exp04_robust_historical_threshold)
        cp=int(pred.exp05_cp_C_historical_probability>=pred.exp05_cp_C_historical_threshold)
        group=['both_missed','cp_only_detected','lab_only_detected','both_detected'][2*lab+cp] if pred.label else ['both_clear','cp_only_fp','lab_only_fp','shared_fp'][2*lab+cp]
        pix=image[sel].astype(float);span=np.ptp(pix,axis=1);width,height=int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)
        record={'deidentified_id':deid,'fold':int(pred.fold),'label':int(pred.label),'historical_overlap_group':group,
                'hb_g_dl':hb,'cutoff_minus_hb_g_dl':cutoff-hb,'who2024_unadjusted_severity':severity(hb,c.sex),
                'roi_pixels':int(sel.sum()),'roi_fraction_fullframe':float(sel.mean()),'bbox_width':width,'bbox_height':height,
                'bbox_fill_fraction':float(sel.sum()/(width*height)),'connected_components':n-1,
                'largest_component_fraction':float(stats[1:,cv2.CC_STAT_AREA].max()/sel.sum()),
                'image_border_contact':bool(sel[0].any() or sel[-1].any() or sel[:,0].any() or sel[:,-1].any()),
                'render_history':row['render_method_recorded'],'stored_profile':row['stored_icc_description'],
                'roi_gray_mean':float(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)[sel].mean()),
                'clipped_any_channel_fraction':float((pix.max(1)==255).mean()),
                'bright_neutral_proxy_fraction':float(((pix.min(1)>=240)&(span<=15)).mean())}
        records.append(record)
        marginx,marginy=max(20,int(width*.12)),max(20,int(height*.4))
        left,right=max(0,xx.min()-marginx),min(image.shape[1],xx.max()+1+marginx)
        top,bottom=max(0,yy.min()-marginy),min(image.shape[0],yy.max()+1+marginy)
        context=image[top:bottom,left:right].copy();inside=sel[top:bottom,left:right]
        boundary=cv2.morphologyEx(inside.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        context[boundary]=[0,255,70]
        thumb=ImageOps.contain(Image.fromarray(context),(290,150))
        tile=Image.new('RGB',(300,185),'#eeeeee');tile.paste(thumb,((300-thumb.width)//2,30+(150-thumb.height)//2))
        ImageDraw.Draw(tile).text((8,8),deid+' approved contour',fill='black');tiles.append(tile)
    frame=pd.DataFrame(records);frame.to_csv(directory/'subject_audit_private.csv',index=False)
    for start in range(0,len(tiles),24):
        canvas=Image.new('RGB',(1200,1110),'white')
        for i,tile in enumerate(tiles[start:start+24]):canvas.paste(tile,((i%4)*300,(i//4)*185))
        canvas.save(directory/f'anatomical_context_private_{start//24+1:02d}.png')
    summary={}
    for group in frame.historical_overlap_group.unique():
        sub=frame[frame.historical_overlap_group==group]
        summary[group]={'n':len(sub),'severity':dict(Counter(sub.who2024_unadjusted_severity)),
                        'hb_g_dl':distribution(sub.hb_g_dl),'cutoff_minus_hb_g_dl':distribution(sub.cutoff_minus_hb_g_dl),
                        'exposure_proxy':distribution(sub.roi_gray_mean),'clipping_proxy':distribution(sub.clipped_any_channel_fraction)}
    numeric={'eligible_images':len(frame),'single_component_masks':int((frame.connected_components==1).sum()),
             'masks_touching_image_edge':int(frame.image_border_contact.sum()),
             'mask_area':distribution(frame.roi_pixels),'bbox_fill_fraction':distribution(frame.bbox_fill_fraction),
             'render_histories':dict(Counter(frame.render_history)),
             'gray_mean':distribution(frame.roi_gray_mean),'clipped_fraction':distribution(frame.clipped_any_channel_fraction),
             'bright_neutral_fraction':distribution(frame.bright_neutral_proxy_fraction)}
    result={'severity_reference':SEVERITY_SOURCE,'severity_rule':'WHO2024 adult15–65: mild >=11 below sex-specific cutoff; moderate8–<11; severe<8 g/dL. Non-pregnancy is owner attestation outside Sheet.',
            'groups':summary,'anatomy_structural':numeric,'repeated_images':repeats,
            'visual_review':'pending agent inspection of five private context sheets; existing owner approvals preserved',
            'limits':'Mask geometry does not determine clinical anatomical adequacy. Clipping/brightness are proxies, not validated glare or acquisition quality. Descriptive Hb groups never change labels. No alternate photo linkage inferred.'}
    write_json(directory/'image_information_audit.json',result)
    return {'subjects':len(frame),'contact_sheets':5,'source_inventory_hash_verified':True,'linked_repeated_eyelid_subjects':len(repeated)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(audit(args.source_root,args.output),indent=2))
