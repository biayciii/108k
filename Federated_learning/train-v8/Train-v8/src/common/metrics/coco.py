"""One COCO evaluator for all architectures; detection metrics only."""
import contextlib
import io

import numpy as np


def detection_metrics(predictions, targets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if len(predictions) != len(targets) or not targets:
        raise ValueError("Evaluation requires equally sized nonempty prediction/target lists")
    annotations, results, images = [], [], []
    for image_id, (pred, truth) in enumerate(zip(predictions, targets), 1):
        images.append({"id": image_id})
        for box, label in zip(truth["boxes"].tolist(), truth["labels"].tolist()):
            x, y, x2, y2 = box
            bbox = [x, y, x2 - x, y2 - y]
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": label, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
        for box, label, score in zip(pred["boxes"].tolist(), pred["labels"].tolist(), pred["scores"].tolist()):
            if label not in (1, 2) or not np.isfinite([*box, score]).all():
                continue
            x, y, x2, y2 = box
            if x2 <= x or y2 <= y:
                continue
            results.append({"image_id": image_id, "category_id": label,
                            "bbox": [x, y, x2 - x, y2 - y], "score": score})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"info": {}, "images": images, "annotations": annotations,
                      "categories": [{"id": 1, "name": "shoulder"}, {"id": 2, "name": "thyroid"}]}
        gt.createIndex()
        if results:
            dt = gt.loadRes(results)
        else:
            dt = COCO()
            dt.dataset = {**gt.dataset, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.iouThrs = np.r_[0.3, np.arange(0.5, 0.96, 0.05)]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]  # all areas, max100 detections

    def mean(values):
        usable = values[values >= 0]
        return float(usable.mean()) if usable.size else None

    return {"map50": mean(precision[1]), "map50_95": mean(precision[1:]),
            "shoulder_ap30": mean(precision[0, :, 0]), "thyroid_ap50": mean(precision[1, :, 1]),
            "n_images": len(images), "metric_type": "COCO 101-point detection AP, maxDets=100"}
