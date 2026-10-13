#!/usr/bin/env python3
"""Exploratory temporal extension, assessed on the same artist-credit-group splits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_models import FEATURES,nested,metrics

ROOT=Path(__file__).resolve().parents[1]
TEMPORAL=['amplitude_time_slope','brightness_time_slope','largest_adjacent_amplitude_rise_db','largest_adjacent_bass_change','onset_rate_time_slope']

def main():
    original=pd.read_csv(ROOT/'data/provisional_92_features.csv').sort_values('Song').reset_index(drop=True)
    new=pd.read_csv(ROOT/'data/recomputed_features.csv').sort_values('Song').reset_index(drop=True)
    assert original.Song.equals(new.Song)
    groups=original.Artist.map(lambda s:' '.join(s.casefold().split())).to_numpy()
    x=np.column_stack([original[FEATURES].to_numpy(float),new[TEMPORAL].to_numpy(float)])
    assert np.isfinite(x).all()
    profiles=pd.read_csv(ROOT/'data/temporal_profiles.csv')
    interior=[]
    for _,row in original.iterrows():
        bins=profiles[profiles.Song==row.Song].sort_values('bin').iloc[1:-1]
        t=(bins.bin.to_numpy()+.5)/10
        rms=bins.rms.to_numpy()
        interior.append([float(np.polyfit(t,rms/(row.rms_mean+1e-8),1)[0]),float(np.polyfit(t,bins.centroid_hz,1)[0]),float(np.diff(20*np.log10(rms+1e-8)).max()),float(np.diff(bins.bass_fraction).max()),float(np.polyfit(t,bins.onsets_per_second,1)[0])])
    xi=np.column_stack([original[FEATURES].to_numpy(float),np.array(interior)])
    boundary_indices=[]
    for song,g in profiles.groupby('Song'):
        db=20*np.log10(g.sort_values('bin').rms.to_numpy()+1e-8)
        boundary_indices.append(int(np.argmax(np.diff(db))))
    audit={'n':len(boundary_indices),'largest_rise_after_first_bin':boundary_indices.count(0),'largest_rise_before_last_bin':boundary_indices.count(8),'boundary_index_counts':pd.Series(boundary_indices).value_counts().sort_index().to_dict(),'interpretation':'Positions near excerpt edges can reflect editing/fades; no musical section labels are inferred.'}
    (ROOT/'results/temporal_boundary_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    original_metrics=json.loads((ROOT/'results/metrics.json').read_text())
    summary={'status':'exploratory extension added after examining primary results; no section annotations or dynamic emotion labels','temporal_features':TEMPORAL,'feature_recomputation_max_abs_difference':{f:float(np.max(np.abs(original[f]-new[f]))) for f in FEATURES},'responses':{},'interior_responses':{},'interior_status':'post-hoc boundary sensitivity; omit first and last bins'}
    pred=original[['Song','Artist','Title','Valence','Arousal']].copy();rows=[]
    for response in ['Valence','Arousal']:
        y=original[response].to_numpy(float)
        run=nested(x,y,groups)
        score=metrics(y,run['pred'])
        score['rmse_change_vs_primary']=score['rmse']-original_metrics['responses'][response]['ridge']['rmse']
        summary['responses'][response]=score
        irun=nested(xi,y,groups)
        summary['interior_responses'][response]=metrics(y,irun['pred'])
        pred[response+'_interior_temporal_pred']=irun['pred']
        pred[response+'_temporal_pred']=run['pred'];pred['fold']=run['folds']
        for seed in range(195,205):
            extended=run if seed==195 else nested(x,y,groups,seed=seed)
            primary=nested(original[FEATURES].to_numpy(float),y,groups,seed=seed)
            ir=irun if seed==195 else nested(xi,y,groups,seed=seed)
            rows.append({'interior_temporal_rmse':metrics(y,ir['pred'])['rmse'],'interior_delta_rmse':metrics(y,ir['pred'])['rmse']-metrics(y,primary['pred'])['rmse'],'response':response,'seed':seed,'primary_rmse':metrics(y,primary['pred'])['rmse'],'temporal_rmse':metrics(y,extended['pred'])['rmse'],'delta_rmse':metrics(y,extended['pred'])['rmse']-metrics(y,primary['pred'])['rmse']})
    pred.to_csv(ROOT/'results/temporal_out_of_fold_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'results/temporal_split_sensitivity.csv',index=False)
    (ROOT/'results/temporal_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
