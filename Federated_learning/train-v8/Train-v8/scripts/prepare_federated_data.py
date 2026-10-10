"""Prepare an explicitly exploratory common dataset from legacy DETR COCO/DICOM/CSV.

No raw identifiers or timestamps are included in the output manifest. Name-derived groups
are proxies, not clinically verified identities. Existing legacy split membership is replaced.
"""
import argparse
import csv
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pydicom
from sklearn.model_selection import GroupShuffleSplit


def name_key(value):
    value = unicodedata.normalize('NFKD', value.casefold().replace('đ', 'd'))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    value = re.sub('[^a-z0-9]', '', value)
    if not value or value in ('nan', 'none', 'unknown'):
        raise ValueError('Missing patient grouping name; supply verified mapping instead')
    return value


def gap_days(row):
    # This format must also agree with the independent gap-hours column on every row.
    fmt = '%m/%d/%Y %H:%M'
    study = datetime.strptime(row['study_datetime'].strip(), fmt)
    treatment = datetime.strptime(row['treatment_datetime'].strip(), fmt)
    hours = (study - treatment).total_seconds() / 3600
    provided = float(row['gap'])
    if not math.isfinite(provided) or hours < 0 or abs(provided - hours) > 1 / 60:
        raise ValueError('Timestamp/gap-hours mismatch; resolve date format or units first')
    return hours / 24


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--allow-name-proxy', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.allow_name_proxy:
        parser.error('This converter needs --allow-name-proxy; its output is exploratory')
    root = args.source.resolve()
    if args.output_dir.exists():
        raise ValueError('Use a new output directory; existing preparations are never overwritten')
    metadata, conflicts = {}, set()
    for split in ('train', 'val', 'test'):
        with (root / 'annotations' / (split + '.csv')).open(encoding='utf-8-sig', newline='') as f:
            for number, row in enumerate(csv.DictReader(f), 2):
                try:
                    value = (name_key(row['name']), gap_days(row))
                except (KeyError, ValueError):
                    raise ValueError(f'Invalid identity/date/gap metadata in {split}.csv row {number}') from None
                uid = row['img_id'].removesuffix('.dcm')
                if uid in metadata and metadata[uid] != value:
                    conflicts.add(uid)
                metadata[uid] = value

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    secret = secrets.token_bytes(32)
    key_path = out / '.pseudonym-key'
    key_path.write_bytes(secret)
    os.chmod(key_path, 0o600)
    token = lambda kind, value: hmac.new(secret, (kind + ':' + value).encode(), hashlib.sha256).hexdigest()
    records, grouping, excluded = [], [], []
    seen_uids, seen_pixels, clipped = set(), set(), 0
    for split in ('train', 'val', 'test'):
        doc = json.loads((root / 'annotations' / (split + '.json')).read_text(encoding='utf-8-sig'))
        categories = {int(c['id']): c['name'].lower() for c in doc['categories']}
        if categories != {1: 'shoulder', 2: 'thyroid'}:
            raise ValueError('Expected COCO categories 1=shoulder and 2=thyroid')
        annotations = defaultdict(list)
        for annotation in doc['annotations']:
            annotations[annotation['image_id']].append(annotation)
        for index, image in enumerate(doc['images']):
            uid = Path(image['file_name']).name.removesuffix('.dcm')
            if uid in seen_uids:
                raise ValueError('Image appears in multiple legacy splits')
            seen_uids.add(uid)
            scan = token('image', uid)
            if uid in conflicts:
                excluded.append({'id': scan, 'reason': 'conflicting_csv_group_or_gap'})
                continue
            if uid not in metadata:
                excluded.append({'id': scan, 'reason': 'no_csv_group_or_gap'})
                continue
            path = (root / 'images' / split / image['file_name']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Image path outside source directory')
            try:
                raw = pydicom.dcmread(path).pixel_array
            except Exception:
                raise ValueError(f'Cannot decode DICOM at {split} image index {index}') from None
            if raw.ndim != 2:
                raise ValueError(f'Expected one 2D DICOM plane at {split} image index {index}')
            # Match the legacy thyroid.py::body_cut coordinate space.
            raw = raw if raw.shape == (512, 512) else raw[:256, :256]
            if raw.shape != (int(image['height']), int(image['width'])):
                raise ValueError('Cropped image dimensions differ from COCO annotation space')
            raw = np.ascontiguousarray(raw, dtype=np.float32)
            if not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError('Invalid pixel data')
            digest = hashlib.sha256(raw.tobytes() + str(raw.shape).encode()).hexdigest()
            if digest in seen_pixels:
                raise ValueError('Duplicate cropped pixels detected; reconcile groups before splitting')
            seen_pixels.add(digest)
            h, w = raw.shape
            boxes, labels = [], []
            for a in annotations[image['id']]:
                if a.get('iscrowd', 0):
                    raise ValueError('Crowd annotation needs explicit handling')
                x, y, bw, bh = map(float, a['bbox'])
                if not np.isfinite([x, y, bw, bh]).all() or bw <= 0 or bh <= 0:
                    raise ValueError('Invalid source bbox')
                original = [x, y, x + bw, y + bh]
                box = [max(0, min(w, x)), max(0, min(h, y)),
                       max(0, min(w, x + bw)), max(0, min(h, y + bh))]
                clipped += box != original
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError('BBox empty after clipping to source image')
                if a['category_id'] not in (1, 2):
                    raise ValueError('Unknown annotation category')
                boxes.append(box)
                labels.append(int(a['category_id']))
            if not boxes:
                raise ValueError('Image has no usable annotation')
            name, gap = metadata[uid]
            relative = 'images/' + scan + '.npy'
            np.save(out / relative, raw, allow_pickle=False)
            records.append({'id': scan, 'patient_id': token('group', name),
                            'image': relative, 'boxes': boxes, 'labels': labels,
                            'gap_days': gap, 'split': 'train'})
            grouping.append(name)

    groups = np.array(grouping)
    train, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.30,
                                         random_state=args.seed).split(records, groups=groups))
    val_local, test_local = next(GroupShuffleSplit(n_splits=1, test_size=0.50,
                                                  random_state=args.seed).split(rest, groups=groups[rest]))
    for i in rest[val_local]:
        records[int(i)]['split'] = 'val'
    for i in rest[test_local]:
        records[int(i)]['split'] = 'test'
    document = {'schema': 'thyroid-detection-v1', 'patient_identity_verified': False,
                'grouping': 'HMAC of normalized CSV-name proxy, not verified clinical identity',
                'split_protocol': 'GroupShuffleSplit 70/15/15 by proxy group; new exploratory split',
                'seed': args.seed, 'gap_definition': '(study_datetime - treatment_datetime) / 86400',
                'source_gap_unit': 'hours; checked against timestamps for all CSV rows',
                'records': records}
    summary = {'source_images': len(seen_uids), 'prepared_images': len(records),
               'excluded_images': len(excluded), 'bbox_clipped': clipped,
               'patient_identity_verified': False, 'source_gap_unit': 'hours',
               'gap_days_range': [min(r['gap_days'] for r in records), max(r['gap_days'] for r in records)],
               'splits': {s: {'images': sum(r['split'] == s for r in records),
                              'proxy_groups': len({r['patient_id'] for r in records if r['split'] == s})}
                          for s in ('train', 'val', 'test')}}
    write_json(out / 'manifest.json', document)
    write_json(out / 'preparation_summary.json', summary)
    write_json(out / 'excluded.json', excluded)
    print(json.dumps(summary, indent=2))
    print('PREPARATION COMPLETED:', out / 'manifest.json')


if __name__ == '__main__':
    main()
