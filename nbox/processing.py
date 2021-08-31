import torch
from PIL import Image
import numpy as np

from torchvision.transforms import functional as F


# def image_processing(input_path):
#     img = Image.open(input_path).convert("RGB")
#     img = torch.tensor(np.asarray(img))
#     img = img.permute(2, 0, 1).unsqueeze(0)
#     return img.float()


def totensor(image):
    min_dim = min(image.size)
    img = F.to_tensor(image)
    img = F.center_crop(img, (min_dim, min_dim))
    img = F.normalize(img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    return img.unsqueeze(0)
