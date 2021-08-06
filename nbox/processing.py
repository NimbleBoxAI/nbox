import torch
from PIL import Image
import numpy as np


def image_processing(input_path):
    img = Image.open(input_path).convert("RGB")
    img = torch.tensor(np.asarray(img))
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img.float()

def totensor(image):
    img = torch.tensor(np.array(image))
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img.float()
