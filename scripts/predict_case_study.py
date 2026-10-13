#!/usr/bin/env python3
"""Predict external feature rows using the saved full-sample refits; no fabricated mood labels."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True,type=Path)
    p.add_argument('--models',type=Path,default=Path('results/fitted_models.json'))
    p.add_argument('--out',type=Path,default=Path('results/case_study_predictions.csv'))
    args=p.parse_args()
    rows=pd.read_csv(args.features);models=json.loads(args.models.read_text());out=[]
    for _,r in rows.iterrows():
        for response,m in models.items():
            if 'Song' in rows and str(r.Song) in m['training_song_ids']:
                raise ValueError('Case study is a training song. Use held-out predictions for study-sample examples.')
            x=r[m['features']].to_numpy(float)
            if not np.isfinite(x).all():raise ValueError('Invalid features')
            c=(x-np.array(m['mean']))/np.array(m['scale'])*np.array(m['coef'])
            pred=float(m['intercept']+c.sum())
            for f,val,con,low,high in zip(m['features'],x,c,m['training_min'],m['training_max']):
                out.append({'Song':r.get('Song','external'),'Title':r.get('Title',''),'response':response,'prediction':pred,'intercept':m['intercept'],'feature':f,'value':val,'contribution':float(con),'outside_training_feature_range':bool(val<low or val>high),'prediction_outside_label_scale':bool(pred<0 or pred>1),'artist_credit_seen_in_training':bool(' '.join(str(r.get('Artist','')).casefold().split()) in m.get('training_artist_credit_groups',[]))})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out).to_csv(args.out,index=False)

if __name__=='__main__':main()
