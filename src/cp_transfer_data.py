"""Strict CP Version2 provenance, alpha-aware tissue inputs and cohort gates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from .local_eyelid_validation import sha256,perceptual_hash,validate_arrays
from .local_eyelid_hybrid_features import manual_roi_tensor,validated_inputs
from .eyelid_transfer_data import local_pixels
from .eyelid_color_data import color_features

SEED=20261012
PREPROCESSING={'version':'opaque_alpha_or_manual_mask_shared_bbox_mean_letterbox224_v1',
    'public_color':'Untagged RGB explicitly assumed sRGB for computation only; unknown camera calibration. No ICC invented.',
    'local_color':'Reuse Exp3 actual ICC-to-sRGB conversion and verified identical DNG-preview profile handling; no camera calibration claim.',
    'alpha':'Opaque alpha==255 core only; transparent and partially composited boundary pixels excluded from model. Public alpha is not a predictor. Local manual0/255 mask follows the same opaque rule.',
    'neural_geometry':'Same native binary tissue selection, tight bounding box, outside tissue ImageNet RGB mean BEFORE bilinear antialiased aspect-preserving resize and centered224 mean letterbox; no fourth channel or dataset-specific fill.',
    'orientation':'Preserve registered raster orientation; train-only horizontal flips, no hue/color jitter or geometric interpolation of clinical labels.',
    'classical_color':'Exact preserved Exp4 A robust18 from original full-resolution manual masks, independently retained from managed neural pixels.',
    'glare':'Retain anatomical glare in both domains; bright-neutral proxy is diagnostic only. No new glare/segmentation masks.',
    'normalization':'Fixed ImageNet RGB mean/std; no learned image color transform, enhancement, equalization or calibration.'}


def snapshot(path: Path) -> dict:
    return {'path':str(path.resolve()),'sha256':sha256(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}


def tissue(rgb: np.ndarray,alpha: np.ndarray) -> tuple[torch.Tensor,np.ndarray]:
    if rgb.dtype!=np.uint8 or alpha.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or alpha.shape!=rgb.shape[:2]:
        raise ValueError('Invalid native RGB/alpha geometry')
    mask=(alpha==255).astype(np.uint8)*255
    validate_arrays(rgb,mask)
    return manual_roi_tensor(rgb,mask),mask


def verify_official_v2(root: Path,archive: Path,expected_archive_sha256: str) -> dict:
    if sha256(archive)!=expected_archive_sha256:raise ValueError('Official archive hash mismatch')
    matched=0
    with zipfile.ZipFile(archive) as z:
        books=[n for n in z.namelist() if Path(n).name=='Cp-Anaemic Dataset.xlsx' and '__MACOSX' not in n]
        if len(books)!=1:raise ValueError('Ambiguous official metadata workbook')
        prefix=Path(books[0]).parent
        for name in z.namelist():
            if name.endswith('/') or '__MACOSX' in Path(name).parts or Path(name).name=='.DS_Store':continue
            rel=Path(name).relative_to(prefix)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive path')
            path=root/rel
            if not path.is_file() or sha256(path)!=hashlib.sha256(z.read(name)).hexdigest():raise ValueError('Local Version2 differs from official release')
            matched+=1
    return {'official_zip':snapshot(archive),'local_files_exactly_matching_official':matched,'doi':'10.17632/m53vz6b7fx.2','license':'CC BY4.0; author academic/research-only disclaimer; this experiment is academic research'}


def audit_public(root: Path) -> tuple[list[dict],dict]:
    book=root/'Cp-Anaemic Dataset.xlsx';source=snapshot(book)
    sheets=pd.read_excel(book,sheet_name=None,dtype=object)
    if set(sheets)!={'Cp-Anaemic Dataset','Anaemic','Non-Anaemic'}:raise ValueError('Public workbook sheets differ')
    core=sheets['Cp-Anaemic Dataset'];parts=pd.concat([sheets['Anaemic'],sheets['Non-Anaemic']],ignore_index=True)
    if list(core)!=['Image_ID','File_Name','Folder_Label','Hb_g_dl'] or list(parts)!=['image_id','filename','folder_label','hb_g_dl','group_id']:
        raise ValueError('Public metadata schema differs')
    if core.Image_ID.duplicated().any() or parts.image_id.duplicated().any() or set(core.Image_ID)!=set(parts.image_id):raise ValueError('Public ID coverage/uniqueness failure')
    secondary=parts.set_index('image_id');records=[];decoded=[];sources=[source];fingerprints={};aliases=0
    for row in core.sort_values('Image_ID').to_dict('records'):
        sid=str(row['Image_ID']);filename=str(row['File_Name']);folder='Anaemic' if row['Folder_Label']=='Anemic' else 'Non-Anaemic'
        if row['Folder_Label'] not in {'Anemic','Non-anemic'} or filename!=sid+'.png' or Path(filename).name!=filename:raise ValueError('Invalid public label/filename')
        hb=float(row['Hb_g_dl']);label=int(hb<11.)
        if not np.isfinite(hb) or not 0<hb<25 or label!=int(folder=='Anaemic'):raise ValueError('Clinical/folder label mismatch')
        second=secondary.loc[sid]
        if second.filename!=filename or second.folder_label!=row['Folder_Label'] or float(second.hb_g_dl)!=hb:raise ValueError('Workbook sheets disagree')
        group=float(second.group_id)
        if not group.is_integer() or group<=0:raise ValueError('Invalid curation group identifier')
        path=root/folder/filename;alias=root/'All Categories'/filename;image_source=snapshot(path);sources.append(image_source)
        if not alias.exists() or sha256(alias)!=image_source['sha256']:raise ValueError('All Categories copy differs or is missing')
        sources.append(snapshot(alias));aliases+=1
        with Image.open(path) as image:
            if image.mode!='RGBA' or image.info.get('icc_profile') or image.getexif() or getattr(image,'n_frames',1)!=1:
                raise ValueError('Unexpected public rendering/color/frames')
            values=np.asarray(image).copy()
        rgb,alpha=values[:,:,:3],values[:,:,3];tensor,mask=tissue(rgb,alpha)
        digest=hashlib.sha256(str(values.shape).encode()+values.tobytes()).hexdigest();decoded.append(digest)
        fp_rgb=rgb.copy();fp_rgb[alpha==0]=128;fingerprints[sid]=perceptual_hash(fp_rgb)
        records.append({'image_id':sid,'curation_group_id':int(group),'patient_id':'not_provided','filename':filename,'path':str(path.resolve()),'label':label,'hb_g_dl':hb,'source_hb_cell':str(row['Hb_g_dl']),
            'source_sha256':image_source['sha256'],'decoded_rgba_sha256':digest,'height':values.shape[0],'width':values.shape[1],
            'opaque_pixels':int((alpha==255).sum()),'visible_pixels':int((alpha>0).sum()),'partial_alpha_pixels':int(((alpha>0)&(alpha<255)).sum()),
            'alpha_boundary_fraction':float(((alpha>0)&(alpha<255)).sum()/(alpha>0).sum()),'icc':'absent_assumed_srgb_not_calibrated','exif':'absent'})
    expected={(r['filename'], 'Anaemic' if r['label'] else 'Non-Anaemic') for r in records}
    actual={(p.name,p.parent.name) for d in ['Anaemic','Non-Anaemic'] for p in (root/d).glob('*.png')}
    if actual!=expected or len(set(decoded))!=len(records) or len(set(r['source_sha256'] for r in records))!=len(records):raise ValueError('Public missing/extra images or exact duplicates')
    if len(list((root/'All Categories').glob('*.png')))!=len(records):raise ValueError('Unexpected duplicate-copy inventory')
    near=[];keys=sorted(fingerprints)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            distance=int(np.count_nonzero(fingerprints[a]!=fingerprints[b]))
            if distance<=6:near.append([a,b,distance])
    parent={s:s for s in keys}
    def find(s):
        while parent[s]!=s:s=parent[s]
        return s
    grouping={}
    for r in records:
        if r['curation_group_id'] in grouping:parent[find(r['image_id'])]=find(grouping[r['curation_group_id']])
        else:grouping[r['curation_group_id']]=r['image_id']
    for a,b,_ in near:parent[find(b)]=find(a)
    groups=[find(r['image_id']) for r in records];labels=np.array([r['label'] for r in records]);hb=np.array([r['hb_g_dl'] for r in records])
    tr,va=next(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(records,labels,groups))
    if set(np.asarray(groups)[tr])&set(np.asarray(groups)[va]) or set(labels[tr])!={0,1} or set(labels[va])!={0,1}:raise ValueError('Public split leakage/class support failure')
    for r,g in zip(records,groups):r['split_group']=g
    audit={'release':2,'unique_images':len(records),'unique_curation_ids':len(set(r['curation_group_id'] for r in records)),
        'unique_subjects':'not_verified; image/curation IDs are not asserted to be patient IDs','split_groups':len(set(groups)),
        'near_candidate_count':len(near),'near_candidates':near,'duplicate_alias_files_not_loaded':aliases,'source_hashes':{s['path']:s['sha256'] for s in sources},
        'class_counts_0_1':np.bincount(labels,minlength=2).tolist(),'hb_g_dl':{'min':float(hb.min()),'max':float(hb.max()),'mean':float(hb.mean()),'median':float(np.median(hb)),'std':float(hb.std())},
        'hb_descriptive_bands':{'below7':int((hb<7).sum()),'7_to_below10':int(((hb>=7)&(hb<10)).sum()),'10_to_below11':int(((hb>=10)&(hb<11)).sum()),'11_or_above':int((hb>=11).sum())},
        'label_policy':'CP dataset study pediatric Hb<11 g/dL, equality11 non-anemia. Preserve original target; do not describe as current age-specific WHO policy. Separate local adult heads.',
        'age_sex_metadata':'Individual age/sex/hospital/patient identifiers absent from released workbook; dataset-level source describes Ghanaian children6-59 months.',
        'training_indices':tr.tolist(),'validation_indices':va.tolist(),'training_class_counts':np.bincount(labels[tr],minlength=2).tolist(),'validation_class_counts':np.bincount(labels[va],minlength=2).tolist(),
        'group_limit':'Conservative curation+near-image groups prevent screened duplicate leakage, but unavailable true patient linkage prevents a verified patient-level public performance claim.',
        'canonical_workbook_snapshot_before_read':source}
    if sha256(book)!=source['sha256']:raise ValueError('Workbook changed during audit')
    return records,audit


def describe(rgb: np.ndarray,mask: np.ndarray) -> dict:
    chosen=mask==255;pixels=rgb[chosen].astype(float);lab=cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2LAB)[chosen].astype(float);lab[:,0]*=100/255;lab[:,1:]-=128
    hsv=cv2.cvtColor(np.ascontiguousarray(rgb),cv2.COLOR_RGB2HSV)[chosen]
    yy,xx=np.where(chosen)
    return {'tissue_pixels':int(chosen.sum()),'height':rgb.shape[0],'width':rgb.shape[1],'bbox_width':int(xx.max()-xx.min()+1),'bbox_height':int(yy.max()-yy.min()+1),
       'rgb_mean':pixels.mean(0).tolist(),'rgb_std':pixels.std(0).tolist(),'lab_mean':lab.mean(0).tolist(),'lab_std':lab.std(0).tolist(),
       'saturation_mean':float(hsv[:,1].mean()/255),'brightness_mean':float(hsv[:,2].mean()/255),'gray_contrast_std':float(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)[chosen].std()),
       'bright_neutral_proxy_fraction':float(((pixels.min(1)>=240)&(np.ptp(pixels,axis=1)<=15)).mean()),'any_channel_clipped_fraction':float((pixels.max(1)==255).mean())}
