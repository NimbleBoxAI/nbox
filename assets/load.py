#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nbox

model = nbox.load("torchvision/maskrcnn_resnet50_fpn", cache_dir="../tests/__ignore/")
