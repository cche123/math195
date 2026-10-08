#!/usr/bin/env python3
"""Reproduce MERGE genre counts using exact metadata tokens; no EDM ground truth."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from collections import Counter
from zipfile import ZipFile

ALLOW = set("2-Step/British Garage|Acid House|Acid Techno|Ambient Breakbeat|Ambient Techno|Detroit Techno|Electro|Euro-Dance|Experimental Techno|Funky Breaks|Hardcore Techno|House|Jazz-House|Jungle/Drum'n'Bass|Left-Field House|Microhouse|Minimal Techno|Neo-Electro|Progressive House|Progressive Trance|Tech-House|Techno|Techno Bass|Trance".split('|'))
SOURCE = 'https://zenodo.org/records/13939205'
ARCHIVE_MD5 = '167f7ba80ab8eca8eee5a8668498a419'

def tokens(value):
    return {s.strip() for s in (value or '').split(',') if s.strip()}

def read_rows(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if not rows or not {'Song', 'Artist', 'Title', 'Genres', 'Styles', 'Quadrant'} <= rows[0].keys():
        raise ValueError('Expected MERGE audio metadata columns are missing.')
    ids = [r['Song'] for r in rows]
    if any(not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError('Song IDs must be present and unique; do not inflate counts with a many-to-many join.')
    return rows

def classify(row):
    electronic = 'Electronic' in tokens(row['Genres'])
    matched = sorted((tokens(row['Genres']) | tokens(row['Styles'])) & ALLOW)
    return electronic, electronic and bool(matched), matched

def audit(raw, out, archive_verified=False):
    rows = read_rows(raw)
    groups = {'all_audio': rows, 'electronic_tagged': [], 'provisional_dance_candidates': []}
    for row in rows:
        electronic, candidate, matched = classify(row)
        if electronic:
            groups['electronic_tagged'].append(row)
        if candidate:
            groups['provisional_dance_candidates'].append(dict(row, matched_dance_tags=', '.join(matched)))
    summary = {
        'source_record': SOURCE, 'record_version_expected': '1.1.0',
        'metadata_sha256': hashlib.sha256(raw).hexdigest(),
        'official_archive_md5_expected': ARCHIVE_MD5,
        'archive_md5_verified_this_run': archive_verified,
        'counts': {g: len(r) for g, r in groups.items()},
        'quadrants': {g: dict(sorted(Counter(r['Quadrant'] for r in rr).items())) for g, rr in groups.items()},
        'electronic_only_genres': sum(tokens(r['Genres']) == {'Electronic'} for r in rows),
        'duplicate_artist_title_rows': len(rows) - len({(r['Artist'], r['Title']) for r in rows}),
        'candidate_allowlist': sorted(ALLOW),
        'interpretation': 'Electronic is a metadata tag. Dance candidates are an earlier provisional rule, not a verified EDM sample. No manual exclusions are performed here.'
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'counts.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name in ('electronic_tagged', 'provisional_dance_candidates'):
        fields = ['Song', 'Artist', 'Title', 'Genres', 'Styles', 'Quadrant']
        if name == 'provisional_dance_candidates':
            fields.append('matched_dance_tags')
        with (out / (name + '.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(groups[name])
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--metadata', type=Path, help='Official merge_audio_complete_metadata.csv')
    inputs.add_argument('--archive', type=Path, help='Downloaded official MERGE_Audio_Complete.zip; checksum verified')
    p.add_argument('--out', type=Path, default=Path('audit_output'))
    args = p.parse_args()
    if args.archive:
        with args.archive.open('rb') as f:
            checksum = hashlib.file_digest(f, 'md5').hexdigest()
        if checksum != ARCHIVE_MD5:
            p.error('Archive MD5 differs from the expected version 1.1.0. Check the release before comparing counts.')
        with ZipFile(args.archive) as z:
            names = [s for s in z.namelist() if s.endswith('/merge_audio_complete_metadata.csv') or s == 'merge_audio_complete_metadata.csv']
            if len(names) != 1:
                p.error('Expected exactly one complete audio metadata CSV in archive.')
            raw = z.read(names[0])
    else:
        raw = args.metadata.read_bytes()
    result = audit(raw, args.out, bool(args.archive))
    print(json.dumps(result['counts'], indent=2))

if __name__ == '__main__':
    main()
