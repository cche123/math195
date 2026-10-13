#!/usr/bin/env python3
"""Download/check official MERGE archive and extract only requested local audio; no redistribution."""
import argparse,hashlib,urllib.request,zipfile
from pathlib import Path
import pandas as pd
URL='https://zenodo.org/api/records/13939205/files/MERGE_Audio_Complete.zip/content'
MD5='167f7ba80ab8eca8eee5a8668498a419'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('data/raw/MERGE_Audio_Complete.zip'))
    p.add_argument('--selection',type=Path,default=Path('data/provisional_92_features.csv'))
    p.add_argument('--audio-out',type=Path,default=Path('data/audio'))
    args=p.parse_args();args.archive.parent.mkdir(parents=True,exist_ok=True)
    if not args.archive.exists():
        print('Downloading approximately 1.16 GB from the official record.',flush=True)
        partial=args.archive.with_suffix('.download')
        urllib.request.urlretrieve(URL,partial)
        with partial.open('rb') as f:checksum=hashlib.file_digest(f,'md5').hexdigest()
        if checksum!=MD5:raise ValueError('Checksum mismatch; partial download retained for inspection')
        partial.replace(args.archive)
    with args.archive.open('rb') as f:checksum=hashlib.file_digest(f,'md5').hexdigest()
    if checksum!=MD5:raise ValueError('Archive checksum mismatch; inspect release/version')
    ids=pd.read_csv(args.selection).Song.astype(str).tolist()
    args.audio_out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        names={Path(s).stem:s for s in z.namelist() if s.lower().endswith('.mp3')}
        for sid in ids:
            if sid not in names:raise ValueError(f'Audio unavailable for {sid}')
            (args.audio_out/(sid+'.mp3')).write_bytes(z.read(names[sid]))
    print(f'Extracted {len(ids)} recordings into {args.audio_out}')

if __name__=='__main__':main()
