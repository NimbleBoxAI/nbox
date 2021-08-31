#!/usr/bin/env python3

import nbox

model = nbox.load("torchvision/resnet18")  # , cache_dir=cache_dir)

url = model.deploy(
    input_object="https://ichef.bbci.co.uk/news/976/cpsprodpb/12A9B/production/_111434467_gettyimages-1143489763.jpg",
    username="",
    password="",
)
