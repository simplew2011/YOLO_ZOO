#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from yolo_zoo.models.utils.torch_utils import select_device

from yolo_zoo.apis.inference import init_detector
from yolo_zoo.apis.test import single_gpu_test
from yolo_zoo.dataset.loader.build_dataloader import build_dataloader
from yolo_zoo.utils.config import Config
from yolo_zoo.utils.newInstance_utils import build_from_dict
from yolo_zoo.utils.registry import DATASET

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--config', type=str, default='yolov5_coco.py', help='model config file')
    parser.add_argument('--checkpoint',type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--coco-val-path', default='/disk2/project/coco/annotations/', help='cocoapi val JSON file Path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--half', action='store_true', help='fp16 half precision')
    opt = parser.parse_args()

    print(opt)

    cfg = Config.fromfile(opt.config)
    cfg.data.val.train = False
    val_dataset = build_from_dict(cfg.data.val, DATASET)
    val_dataloader = build_dataloader(val_dataset, data=cfg.data, shuffle=False)
    device = select_device(opt.device)
    # model = init_detector(opt.config, checkpoint=opt.checkpoint, device=device)
    model = init_detector(opt.config, checkpoint=opt.checkpoint, device=device)
    result = single_gpu_test(model, val_dataloader, half=opt.half,conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=opt.merge,
                     save_json=opt.save_json, augment=opt.augment, verbose=opt.verbose,coco_val_path=opt.coco_val_path)

    print(result)
