#!/usr/bin/env python3
"""Figures derived exclusively from saved evaluated outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results';FIG=ROOT/'Report/figures';FIG.mkdir(parents=True,exist_ok=True)
BLUE='#276F95';ORANGE='#C37645';GRAY='#6B7280'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})

def save(fig,name):
    temporary=FIG/(name+'.temporary')
    fig.savefig(temporary,format='pdf',bbox_inches='tight')
    assert temporary.stat().st_size>1000
    temporary.replace(FIG/(name+'.pdf'))
    fig.savefig(temporary,format='png',bbox_inches='tight')
    temporary.replace(FIG/(name+'.png'))
    plt.close(fig)

def main():
    m=json.loads((RESULT/'metrics.json').read_text());oo=pd.read_csv(RESULT/'out_of_fold_predictions.csv')
    coef=pd.read_csv(RESULT/'fold_coefficients.csv');ab=pd.read_csv(RESULT/'feature_family_ablation.csv')
    reps=pd.read_csv(RESULT/'split_sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,response in zip(axes,['Valence','Arousal']):
        values=[m['responses'][response][name]['rmse'] for name in ['training_mean','ols','ridge']]
        bars=ax.bar(['Training mean','OLS','Ridge'],values,color=[GRAY,ORANGE,BLUE],width=.62)
        ax.set(title=response,ylabel='Out-of-fold RMSE (lower is better)',ylim=(0,max(values)*1.22))
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.003,f'{v:.3f}',ha='center')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Seven acoustic features: 92 provisional dance-style excerpts',fontsize=12)
    fig.tight_layout();save(fig,'01_prediction_error')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    colors={'Q1':'#458A66','Q2':'#BC5D52','Q3':'#7663A0','Q4':'#C69732'}
    for ax,response in zip(axes,['Valence','Arousal']):
        for q,c in colors.items():
            s=oo[oo.Quadrant==q];ax.scatter(s[response],s[response+'_pred'],c=c,label=q,s=25,alpha=.75)
        ax.plot([0,1],[0,1],ls='--',lw=1,color=GRAY)
        ax.set(title=f"{response}: R² = {m['responses'][response]['ridge']['r2']:.3f}",xlabel='Supplied label',ylabel='Held-out ridge prediction',xlim=(.2,1),ylim=(.2,1))
        ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.12)
    fig.tight_layout();save(fig,'02_observed_predicted')
    names={'rms_mean':'Mean amplitude','rms_sd':'Amplitude variation','centroid_hz':'Spectral brightness','bass_power_fraction':'Bass-band fraction','spectral_flux_mean':'Spectral change','onsets_per_second':'Onset rate','tempo_bpm_estimate':'Estimated tempo'}
    fs=list(names)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax,response in zip(axes,['Valence','Arousal']):
        for j,f in enumerate(fs):
            v=coef[(coef.response==response)&(coef.feature==f)].coefficient.to_numpy()
            ax.scatter(v,np.full(len(v),j)+np.linspace(-.1,.1,len(v)),color=BLUE,s=20)
            ax.plot([v.min(),v.max()],[j,j],color=BLUE,lw=1)
        ax.axvline(0,color=GRAY,lw=.8);ax.set(title=response,xlabel='Coefficient per training standard deviation',yticks=range(len(fs)),yticklabels=[names[f] for f in fs]);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();fig.suptitle('Coefficient variation across five outer folds (not confidence intervals)',fontsize=11)
    fig.tight_layout();save(fig,'03_fold_coefficients')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    families=list(m['families'])
    for ax,response in zip(axes,['Valence','Arousal']):
        for j,f in enumerate(families):
            s=ab[(ab.response==response)&(ab.family==f)]
            ax.scatter(s.delta_rmse,np.full(len(s),j)+np.linspace(-.1,.1,len(s)),color=GRAY,s=16,alpha=.65)
            ax.scatter(s.aggregate_delta_rmse.iloc[0],j,color=BLUE,s=65,marker='D',zorder=4)
        ax.axvline(0,color=GRAY,lw=.8);ax.set(title=response,xlabel='RMSE without family minus full-model RMSE',yticks=range(len(families)),yticklabels=families);ax.grid(axis='x',alpha=.12)
    axes[0].invert_yaxis();fig.suptitle('Removing feature families: positive change means worse prediction',fontsize=11)
    fig.tight_layout();save(fig,'04_feature_family_ablation')
    corr=pd.read_csv(RESULT/'descriptive_correlations.csv',index_col=0).loc[fs,['Valence','Arousal']]
    fig,ax=plt.subplots(figsize=(5,4.5));im=ax.imshow(corr.to_numpy(),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set(xticks=[0,1],xticklabels=['Valence','Arousal'],yticks=range(len(fs)),yticklabels=[names[f] for f in fs],title='Descriptive Pearson correlations')
    for i in range(len(fs)):
        for j in range(2):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Correlation');fig.tight_layout();save(fig,'05_descriptive_correlations')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,response in zip(axes,['Valence','Arousal']):
        s=reps[reps.response==response]
        ax.scatter(s.seed,s.rmse_improvement,color=BLUE,s=35)
        ax.axhline(0,color=GRAY,lw=.8)
        ax.set(title=response,xlabel='Grouped split seed',ylabel='Baseline RMSE minus ridge RMSE');ax.grid(alpha=.12)
    fig.suptitle('Split sensitivity: positive values favor ridge (10 overlapping evaluations)',fontsize=11)
    fig.tight_layout();save(fig,'06_split_sensitivity')

if __name__=='__main__':main()
