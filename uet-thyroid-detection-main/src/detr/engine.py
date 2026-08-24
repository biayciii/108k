# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import glob
import math
import os
import sys
from typing import Iterable
import json
import numpy as np

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.map_eval import MapEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from torchmetrics import MeanMetric
from datasets.thyroid import get_im_from_dcm, body_cut
from util.postprocessing import cal_uptake, calc_rsi_acc

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, label2iou_thrs, device, data_path, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    rsi_acc_metric = MeanMetric()
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator = None
    map_evaluator = MapEvaluator(label2iou_thrs=label2iou_thrs)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.3, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        
        # Sorting for caculating rsi accuracy
        for output in results:
            desc_order = output['scores'].argsort(descending=True)
            for k, v in output.items():
                output[k] = v[desc_order]
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # with open('logs.json', 'a') as f:
        #     f.write(str({str(k): v for k, v in res.items()}))
        #     f.write(',\n')
        
        # Caculate thyroid rsi accuracy and shoulder map30
        for img_id, result_dict in res.items():
                gt_anns = base_ds.loadAnns(base_ds.getAnnIds(img_id))
                img_info = base_ds.loadImgs(img_id)[0]
                
                ground_truth_dict = {
                    'boxes': np.asarray([ann['bbox'] for ann in gt_anns]),
                    'labels': np.asarray([ann['category_id'] for ann in gt_anns])
                }
                ground_truth_dict['boxes'][:, 2:] += ground_truth_dict['boxes'][:, :2]
                fp = glob.glob(f'{data_path}/**/{img_info["file_name"]}', recursive=True)[0]
                img = body_cut(get_im_from_dcm(fp))
                
                pred_dict = {
                    'boxes': result_dict['boxes'].cpu().numpy(),
                    'labels':result_dict['labels'].cpu().numpy(),
                    'scores':result_dict['scores'].cpu().numpy()
                }
                map_evaluator.update(ground_truth_dict, pred_dict)
                
                if len(set(ground_truth_dict['labels'])) < 2:
                    continue
                
                gt_lbs = ground_truth_dict['labels']
                ground_truth_dict.update(**{'uptakes': np.asarray([cal_uptake(img, box) for box in ground_truth_dict['boxes']])})
                gt_rsi = ground_truth_dict['uptakes'][gt_lbs == 2] / ground_truth_dict['uptakes'][gt_lbs == 1]
                pred_dict.update(**{'uptakes': np.asarray([cal_uptake(img, box) for box in pred_dict['boxes']])})
                rsi_acc_metric.update(calc_rsi_acc(pred_dict, gt_rsi, eps=0.3, top_k=2))
            
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
    # gather the stats from all processes
    eval_map = map_evaluator.caculate()
    eval_rsi_acc = rsi_acc_metric.compute().item()
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    stats.update(**{'thyroid_mAP@.5': eval_map[2], 'shoulder_mAP@.3': eval_map[1], 'overall_mAP': eval_map['all'], 'rsi_acc': eval_rsi_acc})
    map_evaluator.reset()
    rsi_acc_metric.reset()
    return stats, coco_evaluator
