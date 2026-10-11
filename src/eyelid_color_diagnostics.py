"""Exploratory descriptive diagnostics; never alters source pixels or labels."""
from pathlib import Path
import json,hashlib,io,subprocess
from collections import Counter
import numpy as np
import pandas as pd
import cv2
from PIL import Image,ImageCms,ImageDraw,ImageOps
from .local_eyelid_hybrid_features import validated_inputs
from .local_eyelid_classifier import FEATURES,extract_masked_lab
from .local_eyelid_validation import load_manual_pair, source_snapshot
from .eyelid_transfer_data import local_pixels,dng_preview


def diagnose(source_root: Path, output: Path, preflight: Path) -> dict:
    root=source_root.resolve();base=root/'data/processed';out=output.resolve()
    if not out.is_relative_to(base) or subprocess.run(['git','check-ignore','-q',str(out/'private.csv')],cwd=root).returncode:
        raise ValueError('Diagnostic output must be private and Git-ignored')
    canonical=root/'data/raw/local/metadata/frozen_2026-10-07/AIMA Patients Metadata - Sheet1 values.csv'
    snap=source_snapshot(canonical,canonical.parent/'provenance.json')
    audit=json.loads((preflight/'fresh_local_audit.json').read_text())
    if snap['sha256'] != audit['metadata_source']['sha256']:raise ValueError('Diagnostic audit snapshot is stale')
    if out.exists(): raise FileExistsError('Never overwrite diagnostic output')
    out.mkdir(mode=0o700)
    data=validated_inputs(root,base/'local_eyelid_classification_2026-10-10_v1',base/'local_eyelid_readiness_2026-10-10_v2')
    clinical=pd.DataFrame(json.loads((preflight/'fresh_eligibility.json').read_text())).set_index('subject_id')
    if not np.array_equal(clinical.loc[data['subjects'],'derived_anemia_label'].astype(int),data['labels']):raise ValueError('Clinical labels differ')
    oof=pd.read_csv(base/'local_eyelid_classification_exp03_transfer_2026-10-11_v3/oof_predictions.csv',dtype={'subject_id':str}).set_index('subject_id').loc[data['subjects']]
    assert np.array_equal(oof.label,data['labels']) and np.array_equal(oof.fold,data['folds'])
    records=[];tiles=[];identity=[]
    for index,sid in enumerate(data['subjects']):
     row=data['rows'][sid];raw,mask=load_manual_pair(root/row['annotation_image_path'],root/row['mask_path']);sel=mask>0;yy,xx=np.where(sel)
     managed,managed_mask,color=local_pixels(root,row);assert np.array_equal(mask,managed_mask)
     np.testing.assert_allclose([extract_masked_lab(raw,mask)[f] for f in FEATURES],data['lab'][index],rtol=0,atol=1e-14)
     c=clinical.loc[sid];hb=float(c.normalized_hb_g_per_dl);cutoff=13 if c.sex=='M' else 12
     p=float(oof.loc[sid,'lab_nested_selected_probability']);t=float(oof.loc[sid,'lab_nested_selected_threshold']);y=int(data['labels'][index]);z=int(p>=t)
     group=('TP' if z else 'FN') if y else ('FP' if z else 'TN')
     severity='non_anemia' if not y else ('mild' if hb>=11 else 'moderate' if hb>=8 else 'severe')
     pix=raw[sel].astype(float);gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY);inner=cv2.erode(sel.astype(np.uint8),np.ones((3,3),np.uint8))>0
     lap=cv2.Laplacian(gray,cv2.CV_64F);gradx=cv2.Sobel(gray,cv2.CV_64F,1,0);grady=cv2.Sobel(gray,cv2.CV_64F,0,1)
     span=np.ptp(pix,axis=1);deid=f'E04-{index+1:03d}'
     source=root/row['source_image_path'];original_profile='unknown';preview_match=None
     if source.suffix.lower()=='.dng':
      preview,profile=dng_preview(source);original_profile=ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip();preview_match=bool(np.array_equal(raw,preview))
     else:
      result=subprocess.run(['sips','-g','profile',str(source)],capture_output=True,text=True,check=True)
      original_profile=result.stdout.split('profile:')[-1].strip() if 'profile:' in result.stdout else 'not_recorded'
     record={'deidentified_id':deid,'group':group,'label':y,'fold':int(data['folds'][index]),'probability':p,'selected_threshold':t,'hb_g_dl':hb,'threshold_g_dl':cutoff,'distance_below_threshold_g_dl':cutoff-hb,'severity_unadjusted_who2024':severity,'sex':c.sex,'age':int(c.age),'source_format':source.suffix.lower(),'render_history':row['render_method_recorded'],'stored_profile':row['stored_icc_description'],'original_profile':original_profile,'original_preview_matches_tiff':preview_match,**color,'roi_pixels':int(sel.sum()),'roi_fraction_fullframe':float(sel.mean()),'roi_bbox_width':int(xx.max()-xx.min()+1),'roi_bbox_height':int(yy.max()-yy.min()+1),'roi_centroid_x_relative':float(xx.mean()/raw.shape[1]),'roi_centroid_y_relative':float(yy.mean()/raw.shape[0]),'roi_bbox_fill_fraction':float(sel.sum()/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))),'roi_laplacian_variance':float(lap[inner].var()),'roi_gradient_mean':float(np.hypot(gradx[inner],grady[inner]).mean()),'roi_gray_mean':float(gray[sel].mean()),'roi_gray_std':float(gray[sel].std()),'bright_neutral_proxy_fraction':float(((pix.min(axis=1)>=240)&(span<=15)).mean()),'clipped_any_channel_fraction':float((pix.max(axis=1)==255).mean()),'roi_median_R_over_G':float(np.median((pix[:,0]+1)/(pix[:,1]+1))),'roi_median_B_over_G':float(np.median((pix[:,2]+1)/(pix[:,1]+1))),**{f'original_{f}':float(data['lab'][index,j]) for j,f in enumerate(FEATURES)},**{f'managed_{f}':v for f,v in extract_masked_lab(managed,mask).items()}}
     records.append(record);identity.append({'deidentified_id':deid,'source_subject_id':sid})
     if y:
      crop=raw[yy.min():yy.max()+1,xx.min():xx.max()+1].copy();inside=sel[yy.min():yy.max()+1,xx.min():xx.max()+1];crop[~inside]=30
      thumb=Image.fromarray(crop);thumb.thumbnail((330,140));tile=Image.new('RGB',(350,180),'#eeeeee');tile.paste(thumb,((350-thumb.width)//2,28+(140-thumb.height)//2));ImageDraw.Draw(tile).text((8,8),f'{deid} {group} - original stored color',fill='black');tiles.append(tile)
     if (index+1)%20==0: print(json.dumps({'diagnostic_images':index+1}),flush=True)
    frame=pd.DataFrame(records);assert frame.group.value_counts().to_dict()=={'TN':84,'FP':16,'FN':12,'TP':7}
    frame.to_csv(out/'diagnostic_subjects_private.csv',index=False);pd.DataFrame(identity).to_csv(out/'identity_key_private.csv',index=False)
    num=['hb_g_dl','distance_below_threshold_g_dl','roi_pixels','roi_fraction_fullframe','roi_laplacian_variance','roi_gradient_mean','roi_gray_mean','bright_neutral_proxy_fraction','clipped_any_channel_fraction','original_mean_L','original_mean_a','original_mean_b']
    summary={}
    for group in ['FN','TP','non_anemia']:
     sub=frame[frame.group.isin(['FP','TN'])] if group=='non_anemia' else frame[frame.group==group]
     summary[group]={'n':len(sub),'severity':sub.severity_unadjusted_who2024.value_counts().to_dict(),'render_histories':sub.render_history.value_counts().to_dict(),'sex':sub.sex.value_counts().to_dict(),'color_corrected_subjects':int((sub.color_changed_pixels>0).sum()),'statistics':{f:{'mean':float(sub[f].mean()),'median':float(sub[f].median()),'q25':float(sub[f].quantile(.25)),'q75':float(sub[f].quantile(.75)),'min':float(sub[f].min()),'max':float(sub[f].max())} for f in num}}
    with (out/'diagnostic_summary.json').open('x') as f:json.dump(summary,f,indent=2)
    canvas=Image.new('RGB',(350*4,180*5),'white')
    for i,tile in enumerate(tiles):canvas.paste(tile,((i%4)*350,(i//4)*180))
    canvas.save(out/'anemia_roi_contact_sheet_private.png')
    lines=['# Experiment 4 pretraining descriptive diagnostic','', 'Exploratory review of the preserved LAB nested-threshold predictions; no fitting or tuning. All 119 subjects, verified labels and manual anatomical masks are unchanged. This output is private. A separate identity key links pseudonyms to source subjects.','', 'Severity uses the unadjusted WHO 2024 adult 15–65 table: mild Hb >=11 g/dL below sex-specific cutoff, moderate 8–<11, severe <8. Non-pregnancy is owner-attested outside the Sheet. https://www.ncbi.nlm.nih.gov/books/NBK602183/table/fm-ch1.tab2/','', '| Group | n | Hb median (range), g/dL | Cutoff minus Hb median | Severity |','| --- | ---: | --- | ---: | --- |']
    for group,s in summary.items():
     h=s['statistics']['hb_g_dl'];delta=s['statistics']['distance_below_threshold_g_dl']['median'];lines.append(f"| {group} | {s['n']} | {h['median']:.2f} ({h['min']:.2f}–{h['max']:.2f}) | {delta:.2f} | {s['severity']} |")
    lines+=['','## Quality and rendering diagnostics','', 'These are descriptive proxies, not accepted quality exclusions or glare masks. Native-resolution ROI Laplacian/gradient values depend on resolution, tissue texture and rendering; they cannot alone establish focus. Bright-neutral pixels are an unvalidated specular-highlight proxy and can represent naturally pale tissue. ROI R/G and B/G reflect tissue color as well as capture and cannot establish white balance without a reference. All glare remains in every model input. Anatomical tissue coverage is owner-reviewed; size/location metrics cannot establish clinical adequacy.','', '| Quantity (median) | FN | TP | Non-anemia |','| --- | ---: | ---: | ---: |']
    for key in num[2:]:lines.append('| '+key+' | '+' | '.join(f"{summary[g]['statistics'][key]['median']:.6g}" for g in ['FN','TP','non_anemia'])+' |')
    lines+=['','## Identified missed subjects (private pseudonyms)','', '| ID | Hb g/dL | Cutoff minus Hb | Severity | Fold | Score | Cutoff |','| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in records:
     if r['group']=='FN':lines.append(f"| {r['deidentified_id']} | {r['hb_g_dl']:.2f} | {r['distance_below_threshold_g_dl']:.2f} | {r['severity_unadjusted_who2024']} | {r['fold']} | {r['probability']:.4f} | {r['selected_threshold']:.4f} |")
    lines+=['','## Limitations and supported investigations','', 'The 12 missed cases are not an independent tuning set. Group differences are conditioned on predictions learned from these same development subjects. Seven detected cases are too few for reliable causal or subgroup conclusions. Hb distances describe borderline labels without changing them. Compare managed color with original features to isolate ICC effects; compare limited robust/region features to investigate whether averages discard useful tissue-color variation. Do not exclude or relabel difficult subjects. No poor-quality causal explanation is established from these proxies. Missing color-chart and capture-standardization evidence prevents accurate retrospective camera calibration.','', 'The private contact sheet shows all 19 approved anatomical crops in original stored color, for visual review without source filenames or direct identifiers.']
    with (out/'diagnostic_report_private.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'groups':frame.group.value_counts().to_dict(),'summary':summary,'profile_corrections':int((frame.color_changed_pixels>0).sum())},allow_nan=False))

    return summary
