#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-*- coding:utf-8 -*-
import random
imgsz = 640
imgs = []
sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
sf = sz / max(imgs.shape[2:])  # scale factor
if sf != 1:
    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)