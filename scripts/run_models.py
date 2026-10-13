#!/usr/bin/env python3
"""Nested artist-credit-group evaluation of the prespecified seven-feature ridge models."""
import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

FEATURES = ['rms_mean','rms_sd','centroid_hz','bass_power_fraction','spectral_flux_mean','onsets_per_second','tempo_bpm_estimate']
FAMILIES = {'Amplitude':['rms_mean','rms_sd'], 'Rhythm':['onsets_per_second','tempo_bpm_estimate'], 'Brightness/change':['centroid_hz','spectral_flux_mean'], 'Bass band':['bass_power_fraction']}
GRID = np.logspace(-4, 4, 9)

@dataclass
class Fit:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    penalty: float
    def predict(self, x):
        return self.intercept + ((x-self.mean)/self.scale) @ self.coef
    def contributions(self, x):
        return ((x-self.mean)/self.scale) * self.coef

def fit_ridge(x, y, penalty):
    """Report loss: mean squared residual + lambda * squared slopes; alpha=m*lambda."""
    mean=x.mean(axis=0)
    scale=x.std(axis=0, ddof=0)
    active=scale>0
    safe=np.where(active,scale,1.0)
    z=(x-mean)/safe
    coef=np.zeros(x.shape[1])
    if active.any():
        model=Ridge(alpha=len(y)*float(penalty),fit_intercept=True,solver='svd')
        model.fit(z[:,active],y)
        coef[active]=model.coef_
        intercept=float(model.intercept_)
    else:
        intercept=float(y.mean())
    return Fit(mean,safe,coef,intercept,float(penalty))

def splits(groups, folds, seed):
    if len(np.unique(groups)) < folds:
        raise ValueError('Insufficient unique groups for requested folds.')
    cv=GroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    result=list(cv.split(np.zeros(len(groups)),groups=groups))
    for train,test in result:
        assert not set(groups[train]) & set(groups[test]), 'Group leakage'
    return result

def tune(x,y,groups,folds,seed):
    partitions=splits(groups,folds,seed)
    errors=[]
    for penalty in GRID:
        sse=0.;n=0
        for train,valid in partitions:
            model=fit_ridge(x[train],y[train],penalty)
            sse+=float(np.square(y[valid]-model.predict(x[valid])).sum());n+=len(valid)
        errors.append(sse/n)
    # GRID is ascending; np.argmin uses the smaller penalty on an exact tie.
    index=int(np.argmin(errors))
    return float(GRID[index]),errors

def metrics(y,pred):
    residual=y-pred
    sst=float(np.square(y-y.mean()).sum())
    return {'rmse':float(np.sqrt(np.mean(residual**2))), 'mae':float(np.mean(np.abs(residual))), 'r2':float(1-float(np.square(residual).sum())/sst) if sst else None, 'out_of_range':int(((pred<0)|(pred>1)).sum())}

def nested(x,y,groups,outer=5,inner=3,seed=195):
    pred=np.full(len(y),np.nan);base=pred.copy();ols=pred.copy()
    contrib=np.full(x.shape,np.nan);fold_ids=np.full(len(y),-1)
    foldrows=[];fits=[]
    for fold,(train,test) in enumerate(splits(groups,outer,seed),1):
        penalty,losses=tune(x[train],y[train],groups[train],inner,seed+fold)
        fit=fit_ridge(x[train],y[train],penalty)
        pred[test]=fit.predict(x[test]);base[test]=y[train].mean()
        ols[test]=fit_ridge(x[train],y[train],0).predict(x[test])
        contrib[test]=fit.contributions(x[test]);fold_ids[test]=fold
        assert np.allclose(pred[test],fit.intercept+contrib[test].sum(axis=1))
        foldrows.append({'fold':fold,'train_n':len(train),'test_n':len(test),'train_groups':len(set(groups[train])),'test_groups':len(set(groups[test])),'lambda':penalty,'alpha':len(train)*penalty,'rmse':metrics(y[test],pred[test])['rmse'],'baseline_rmse':metrics(y[test],base[test])['rmse'],'inner_mse':losses})
        fits.append(fit)
    assert np.isfinite(pred).all() and (fold_ids>0).all()
    return {'pred':pred,'baseline':base,'ols':ols,'contributions':contrib,'folds':fold_ids,'foldrows':foldrows,'fits':fits}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('data/provisional_92_features.csv'))
    ap.add_argument('--out',type=Path,default=Path('results'))
    ap.add_argument('--seed',type=int,default=195)
    ap.add_argument('--repeats',type=int,default=10)
    args=ap.parse_args()
    df=pd.read_csv(args.input).sort_values('Song',kind='stable').reset_index(drop=True)
    needed=set(FEATURES+['Song','Artist','Title','Valence','Arousal','Quadrant'])
    if not needed<=set(df):raise ValueError('Missing required columns')
    if df.Song.isna().any() or not df.Song.is_unique:raise ValueError('Song IDs must be present and unique')
    if df.Artist.isna().any() or df.Artist.str.strip().eq('').any():raise ValueError('Artist credits required')
    if not np.isfinite(df[FEATURES+['Valence','Arousal']].to_numpy(float)).all():raise ValueError('Non-finite predictors/responses; inspect extraction instead of silently dropping rows')
    if 'status' in df and not df.status.eq('ok').all():raise ValueError('Failed extraction rows')
    # Full artist-credit strings are grouped, normalized only for whitespace and case.
    groups=df.Artist.map(lambda s:' '.join(s.casefold().split())).to_numpy()
    x=df[FEATURES].to_numpy(float)
    args.out.mkdir(parents=True,exist_ok=True)
    oo=df[['Song','Artist','Title','Quadrant','Valence','Arousal']].copy()
    summary={'sample_status':'provisional dance-style candidates, not manually verified EDM','n':len(df),'artist_credit_groups':len(set(groups)), 'duplicate_artist_title_rows':int(df.duplicated(['Artist','Title']).sum()),'features':FEATURES,'families':FAMILIES,'lambda_grid':GRID.tolist(),'outer_folds':5,'inner_folds':3,'seed':args.seed,'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'quadrants':df.Quadrant.value_counts().sort_index().to_dict(),'responses':{}}
    coefrows=[];ablrows=[];repeatrows=[];foldrows=[];models={}
    for response in ['Valence','Arousal']:
        y=df[response].to_numpy(float)
        full=nested(x,y,groups,seed=args.seed)
        response_summary={name:metrics(y,full[key]) for name,key in [('ridge','pred'),('training_mean','baseline'),('ols','ols')]}
        for key in ['ridge','training_mean','ols']:
            residual=y-full[{'ridge':'pred','training_mean':'baseline','ols':'ols'}[key]]
            artist_mse=pd.DataFrame({'group':groups,'error':residual**2}).groupby('group').error.mean().mean()
            response_summary[key]['equal_group_rmse']=float(np.sqrt(artist_mse))
        summary['responses'][response]=response_summary
        oo[response+'_pred']=full['pred'];oo[response+'_baseline']=full['baseline'];oo[response+'_ols']=full['ols'];oo['fold']=full['folds']
        for j,f in enumerate(FEATURES):
            oo[response+'_contribution_'+f]=full['contributions'][:,j]
        for k,fit in enumerate(full['fits'],1):
            for f,b in zip(FEATURES,fit.coef):coefrows.append({'response':response,'fold':k,'feature':f,'coefficient':float(b),'lambda':fit.penalty})
        for row in full['foldrows']:foldrows.append(dict(response=response,**row))
        for family,drop in FAMILIES.items():
            keep=[j for j,f in enumerate(FEATURES) if f not in drop]
            ab=nested(x[:,keep],y,groups,seed=args.seed)
            for fold in range(1,6):
                ix=full['folds']==fold
                delta=metrics(y[ix],ab['pred'][ix])['rmse']-metrics(y[ix],full['pred'][ix])['rmse']
                ablrows.append({'response':response,'family':family,'fold':fold,'delta_rmse':delta,'aggregate_delta_rmse':metrics(y,ab['pred'])['rmse']-metrics(y,full['pred'])['rmse']})
            oo[response+'_without_'+family.replace('/','_').replace(' ','_')]=ab['pred']
        for repeat in range(args.repeats):
            seed=args.seed+repeat
            run=full if repeat==0 else nested(x,y,groups,seed=seed)
            a=metrics(y,run['pred']);b=metrics(y,run['baseline'])
            repeatrows.append({'response':response,'seed':seed,'ridge_rmse':a['rmse'],'baseline_rmse':b['rmse'],'ridge_r2':a['r2'],'rmse_improvement':b['rmse']-a['rmse']})
        penalty,losses=tune(x,y,groups,5,args.seed+1000)
        final=fit_ridge(x,y,penalty)
        models[response]={'features':FEATURES,'mean':final.mean.tolist(),'scale':final.scale.tolist(),'coef':final.coef.tolist(),'intercept':final.intercept,'lambda':final.penalty,'training_min':x.min(axis=0).tolist(),'training_max':x.max(axis=0).tolist(),'training_song_ids':df.Song.tolist(),'training_artist_credit_groups':sorted(set(groups)),'status':'refit on full provisional sample; evaluate only with saved out-of-fold predictions'}
        print(response,json.dumps(response_summary),flush=True)
    oo.to_csv(args.out/'out_of_fold_predictions.csv',index=False)
    pd.DataFrame(coefrows).to_csv(args.out/'fold_coefficients.csv',index=False)
    pd.DataFrame(ablrows).to_csv(args.out/'feature_family_ablation.csv',index=False)
    pd.DataFrame(repeatrows).to_csv(args.out/'split_sensitivity.csv',index=False)
    (args.out/'fold_details.json').write_text(json.dumps(foldrows,indent=2)+'\n')
    (args.out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.out/'fitted_models.json').write_text(json.dumps(models,indent=2)+'\n')
    df[FEATURES+['Valence','Arousal']].corr().to_csv(args.out/'descriptive_correlations.csv')

if __name__=='__main__':main()
