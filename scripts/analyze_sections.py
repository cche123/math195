#!/usr/bin/env python3
"""Measure manually annotated sections; requires actual start/end times and section labels."""
import argparse
from pathlib import Path
import pandas as pd
import librosa
from extract_features import extract
VALID={'intro','buildup','drop','breakdown','core','outro'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True,type=Path)
    p.add_argument('--audio-dir',required=True,type=Path)
    p.add_argument('--out',type=Path,default=Path('results/annotated_section_features.csv'))
    args=p.parse_args();df=pd.read_csv(args.annotations)
    if not {'Song','start_seconds','end_seconds','section'}<=set(df):raise ValueError('Missing annotation columns')
    index={f.stem:f for f in args.audio_dir.rglob('*') if f.suffix.lower() in {'.mp3','.wav','.flac','.ogg'}}
    outputs=[]
    for song,g in df.groupby('Song'):
        if song not in index:raise FileNotFoundError(song)
        duration=librosa.get_duration(path=index[song]);previous_end=0
        for _,r in g.sort_values('start_seconds').iterrows():
            if r.section not in VALID:raise ValueError('Section must be independently reviewed; unreviewed labels cannot be analyzed')
            start=float(r.start_seconds);end=float(r.end_seconds)
            if start<previous_end or end>duration or end-start<6:raise ValueError('Intervals must not overlap, must lie within audio, and must span at least six seconds')
            previous_end=end
            values,_=extract(index[song],start=start,end=end)
            outputs.append(dict(r.to_dict(),**values))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(outputs).to_csv(args.out,index=False)
    print('Exported measured section features. Song-level mood scores are not section-level emotion annotations.')

if __name__=='__main__':main()
