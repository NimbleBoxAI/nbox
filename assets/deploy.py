#!/usr/bin/env python3

import nbox
import torch


class DoubleHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Linear(2, 4)
        self.f2 = torch.nn.Linear(2, 4)

    def forward(self, x, y):
        return self.f1(x) + self.f2(y)


model = nbox.Model(DoubleHeadModel(), {"x": "image", "y": "image"})
