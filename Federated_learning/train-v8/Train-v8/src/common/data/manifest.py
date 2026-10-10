"""Validated common manifest: one grayscale plane, boxes, patient group, split."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_brightness_stack, percentile_normalize


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_manifest(path, data_root, exploratory=False):
    path, root = Path(path), Path(data_root).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("schema") != "thyroid-detection-v1":
        raise ValueError("Expected schema thyroid-detection-v1; see Train/init.md")
    if document.get("patient_identity_verified") is not True and not exploratory:
        raise ValueError("Patient identities are unverified; use --exploratory only for a development run")
    rows = document.get("records", [])
    if not rows:
        raise ValueError("Manifest has no records")
    ids, patient_splits, images, hashes = set(), {}, set(), {}
    for row in rows:
        if {"name", "birth_year", "dob", "PatientName"} & row.keys():
            raise ValueError("Manifest must omit direct patient identifiers")
        if row["id"] in ids:
            raise ValueError("Duplicate image ID")
        ids.add(row["id"])
        patient, split = row["patient_id"], row["split"]
        if not isinstance(patient, str) or len(patient) != 64 or any(c not in "0123456789abcdef" for c in patient):
            raise ValueError("patient_id must be a pseudonymized SHA256 hex token")
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError("The same patient occurs in more than one split")
        patient_splits[patient] = split
        image_path = (root / row["image"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError("An image is missing or outside data-root")
        if image_path in images:
            raise ValueError("One image file appears more than once")
        images.add(image_path)
        digest = file_sha256(image_path)
        if digest in hashes:
            raise ValueError("Duplicate image content in manifest")
        hashes[digest] = row["id"]
        boxes = np.asarray(row["boxes"], dtype=float).reshape(-1, 4)
        labels = row["labels"]
        if len(boxes) != len(labels) or any(label not in (1, 2) for label in labels):
            raise ValueError("Labels must be 1=shoulder, 2=thyroid with one label per box")
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(boxes < 0):
            raise ValueError("Boxes must be finite positive-area xyxy coordinates")
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")}
    if not all(counts.values()):
        raise ValueError("Manifest must have nonempty train, val and test splits")
    return document, {"sha256": file_sha256(path),
                      "counts": counts, "patients": len(patient_splits),
                      "patient_identity_verified": document.get("patient_identity_verified") is True,
                      "image_hashes": hashes}


class ManifestDataset(Dataset):
    def __init__(self, rows, root, cfg):
        self.rows, self.root, self.cfg = list(rows), Path(root), cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["image"]
        if path.suffix.lower() == ".npy":
            raw = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".dcm":
            import pydicom
            raw = pydicom.dcmread(str(path)).pixel_array
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = raw[0]
            if raw.ndim == 2 and raw.shape != (512, 512):
                cut_size = int(self.cfg["data"].get("cut_size", 256))
                raw = raw[:cut_size, :cut_size]
        else:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2 or min(raw.shape) < 1 or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Images must be finite nonnegative 2D grayscale planes in ROI coordinate space")
        h, w = raw.shape
        expected = (row.get("height"), row.get("width"))
        if all(value is not None for value in expected) and expected != (h, w):
            raise ValueError("Preprocessed DICOM size does not match the COCO annotation size")
        boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
        if len(boxes) and (torch.any(boxes[:, 2] > w) or torch.any(boxes[:, 3] > h)):
            raise ValueError("Bounding box falls outside source image")
        stack = build_brightness_stack(percentile_normalize(raw), self.cfg["data"]["brightness_level"])
        size = self.cfg["data"]["image_size"]
        resized = cv2.resize(stack, (size, size), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        image = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float() / 255
        target = {"boxes": boxes * torch.tensor([size / w, size / h] * 2),
                  "labels": torch.tensor(row["labels"], dtype=torch.int64)}
        return image, target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
