#!/usr/bin/env python
# -*- coding: utf-8 -*-

from yolo_zoo.apis.inference import init_detector
from yolo_zoo.apis.test import single_gpu_test
from yolo_zoo.dataset.loader.build_dataloader import build_dataloader
from yolo_zoo.utils.config import Config
from yolo_zoo.utils.newInstance_utils import build_from_dict
from yolo_zoo.utils.registry import DATASET
# config = '/disk2/project/pytorch-YOLOv4/cfg/yolov4_hand_gpu.py'
config = '/disk2/project/pytorch-YOLOv4/cfg/yolov5_hand_gpu.py'
config = '/disk2/project/pytorch-YOLOv4/cfg/ppyolo_hand_gpu.py'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/yolov4-hand/latest.pth'
checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/ppyolo_hand/latest.pth'
# checkpoint = '/disk2/project/pytorch-YOLOv4/work_dirs/test/latest.pth'
cfg = Config.fromfile(config)
cfg.data.val.train = False
val_dataset = build_from_dict(cfg.data.val, DATASET)
val_dataloader = build_dataloader(val_dataset, data=cfg.data,shuffle=False)

model = init_detector(config, checkpoint=checkpoint, device='cuda:0')

results = single_gpu_test(model, val_dataloader, show=False)
print(results)

