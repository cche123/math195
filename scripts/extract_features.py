#!/usr/bin/env python3
"""Extract full-excerpt features and temporal descriptors from local authorized audio.
Temporal descriptors are acoustic measurements, not validated section or dynamic-emotion labels.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

FEATURES=['rms_mean','rms_sd','centroid_hz','bass_power_fraction','spectral_flux_mean','onsets_per_second','tempo_bpm_estimate']
TEMPORAL=['amplitude_time_slope','brightness_time_slope','largest_adjacent_amplitude_rise_db','largest_adjacent_bass_change','onset_rate_time_slope']

def extract(path,start=None,end=None):
    sr=22050;hop=512;nfft=2048
    y,_=librosa.load(path,sr=sr,mono=True,offset=float(start or 0),duration=(float(end)-float(start or 0)) if end is not None else None)
    duration=len(y)/sr
    if duration<6:raise ValueError('At least six seconds required')
    s=np.abs(librosa.stft(y,n_fft=nfft,hop_length=hop));power=s**2
    freq=librosa.fft_frequencies(sr=sr,n_fft=nfft);band=(freq>=20)&(freq<250)
    rms=librosa.feature.rms(S=s,frame_length=nfft)[0]
    centroid=librosa.feature.spectral_centroid(S=s,sr=sr)[0]
    flux=np.maximum(np.diff(s,axis=1),0).sum(axis=0)
    onset=librosa.onset.onset_strength(S=librosa.power_to_db(power+1e-12),sr=sr,hop_length=hop)
    peaks=librosa.onset.onset_detect(onset_envelope=onset,sr=sr,hop_length=hop)
    tempo=float(librosa.feature.tempo(onset_envelope=onset,sr=sr,hop_length=hop)[0])
    values={'duration_seconds':duration,'rms_mean':float(rms.mean()),'rms_sd':float(rms.std()),'centroid_hz':float(centroid.mean()),'bass_power_fraction':float(power[band].sum()/(power.sum()+1e-12)),'spectral_flux_mean':float(flux.mean()),'onsets_per_second':len(peaks)/duration,'tempo_bpm_estimate':tempo}
    frame_times=librosa.frames_to_time(np.arange(s.shape[1]),sr=sr,hop_length=hop)
    peak_times=librosa.frames_to_time(peaks,sr=sr,hop_length=hop)
    # Ten equal-duration bins retain the order of change while matching excerpt lengths.
    edges=np.linspace(0,duration,11);profiles=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ix=(frame_times>=lo)&(frame_times<hi)
        assert ix.any()
        profiles.append({'bin':j,'start_seconds':lo,'end_seconds':hi,'rms':float(rms[ix].mean()),'centroid_hz':float(centroid[ix].mean()),'bass_fraction':float(power[band][:,ix].sum()/(power[:,ix].sum()+1e-12)),'onsets_per_second':float(((peak_times>=lo)&(peak_times<hi)).sum()/(hi-lo))})
    q=pd.DataFrame(profiles);t=(np.arange(10)+.5)/10
    amplitude_db=20*np.log10(q.rms.to_numpy()+1e-8)
    # Relative amplitude slope avoids treating different overall mastering levels as time trends.
    normalized_amplitude=q.rms.to_numpy()/(values['rms_mean']+1e-8)
    values.update({'amplitude_time_slope':float(np.polyfit(t,normalized_amplitude,1)[0]),'brightness_time_slope':float(np.polyfit(t,q.centroid_hz,1)[0]),'largest_adjacent_amplitude_rise_db':float(np.diff(amplitude_db).max()),'largest_adjacent_bass_change':float(np.diff(q.bass_fraction).max()),'onset_rate_time_slope':float(np.polyfit(t,q.onsets_per_second,1)[0])})
    return values,q

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-dir',required=True,type=Path)
    p.add_argument('--metadata',required=True,type=Path,help='CSV with Song and optional Artist, Title, emotion labels; outputs retain these columns')
    p.add_argument('--out',type=Path,default=Path('data/recomputed_features.csv'))
    args=p.parse_args();metadata=pd.read_csv(args.metadata)
    if not metadata.Song.is_unique:raise ValueError('Duplicate Song IDs')
    index={f.stem:f for f in args.audio_dir.rglob('*') if f.suffix.lower() in {'.mp3','.wav','.flac','.ogg'}}
    records=[];profiles=[]
    for i,(_,row) in enumerate(metadata.iterrows(),1):
        if row.Song not in index:raise FileNotFoundError(f'Audio missing for {row.Song}')
        vals,q=extract(index[row.Song]);record=row.to_dict();record.update(vals);record['status']='ok';records.append(record)
        q.insert(0,'Song',row.Song);profiles.append(q)
        if i%10==0:print(f'Extracted {i}/{len(metadata)}',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(args.out,index=False)
    pd.concat(profiles,ignore_index=True).to_csv(args.out.with_name('temporal_profiles.csv'),index=False)
    args.out.with_suffix('.provenance.json').write_text(json.dumps({'librosa':librosa.__version__,'sample_rate':22050,'n_fft':2048,'hop_length':512,'temporal_bins':10,'section_labels':'none; acoustic profiles only'},indent=2)+'\n')

if __name__=='__main__':main()
