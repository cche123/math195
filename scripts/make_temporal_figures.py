#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'Report/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def main():
    q=pd.read_csv(ROOT/'data/temporal_profiles.csv')
    d=pd.read_csv(ROOT/'data/recomputed_features.csv')
    chosen=d.sort_values(['largest_adjacent_amplitude_rise_db','Song'],ascending=[False,True]).head(2)
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    selected=[]
    for row,(_,r) in enumerate(chosen.iterrows()):
        s=q[q.Song==r.Song];t=(s.start_seconds+s.end_seconds)/2
        amplitude=20*np.log10(s.rms.to_numpy()+1e-8)
        relative=amplitude-amplitude.max();boundary=s.end_seconds.iloc[int(np.argmax(np.diff(amplitude)))]
        series=[relative,s.centroid_hz,s.bass_fraction]
        labels=['Relative amplitude (dB)','Spectral centroid (Hz)','Bass-band power fraction']
        for ax,v,label in zip(axes[row],series,labels):
            ax.plot(t,v,'o-',c='#276F95',ms=4);ax.axvline(boundary,c='#C37645',ls='--',lw=1)
            ax.set(xlabel='Seconds within excerpt',ylabel=label);ax.grid(alpha=.12)
        axes[row,1].set_title(f'{r.Artist}: {r.Title}\n{supplied_text(r)}',fontsize=11)
        selected.append({'Song':r.Song,'Artist':r.Artist,'Title':r.Title,'Quadrant':r.Quadrant,'largest_adjacent_amplitude_rise_db':float(r.largest_adjacent_amplitude_rise_db),'largest_rise_boundary_seconds':float(boundary),'selection':'Two largest adjacent-bin amplitude increases; descriptive examples only; no section annotations'})
    fig.suptitle('Acoustic changes within excerpts: dashed line marks the largest amplitude rise',fontsize=11)
    fig.tight_layout();save(fig,'07_temporal_profiles')
    (ROOT/'results/temporal_examples.json').write_text(json.dumps(selected,indent=2)+'\n')
    m=json.loads((ROOT/'results/metrics.json').read_text());tm=json.loads((ROOT/'results/temporal_metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,response in zip(axes,['Valence','Arousal']):
        vals=[m['responses'][response]['training_mean']['rmse'],m['responses'][response]['ridge']['rmse'],tm['responses'][response]['rmse'],tm['interior_responses'][response]['rmse']]
        bars=ax.bar(['Mean','Seven\nfeatures','Temporal','Trimmed\ntemporal'],vals,color=['#6B7280','#276F95','#64A590','#C37645'])
        ax.set(title=response,ylabel='Held-out RMSE',ylim=(0,max(vals)*1.23))
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.003,f'{v:.4f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    fig.suptitle('Temporal checks: raw and edge-trimmed descriptors on the same outer groups',fontsize=11)
    fig.tight_layout();save(fig,'08_temporal_model_comparison')

def supplied_text(r):return f'Static labels: valence {r.Valence:.2f}, arousal {r.Arousal:.2f}'
def save(fig,name):
    temporary=FIG/(name+'.temporary')
    fig.savefig(temporary,format='pdf',bbox_inches='tight')
    assert temporary.stat().st_size>1000
    temporary.replace(FIG/(name+'.pdf'))
    fig.savefig(temporary,format='png',bbox_inches='tight',dpi=180)
    temporary.replace(FIG/(name+'.png'))
    plt.close(fig)
if __name__=='__main__':main()
