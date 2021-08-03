import torch
from PIL import Image
import numpy as np

class Processing:
    # Define processing methods for different input types, eg video, images,
    # text, etc... so they can be called later in the Models class in the
    # __call__ method.

    # Probably also add something like transformation from torchvision.transforms
    @staticmethod
    def image_processing(input_path):
        img = Image.open(input_path).convert("RGB")
        img = torch.tensor(np.asarray(img))
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img.float()

    @staticmethod
    def totensor(image):
        img = torch.tensor(np.asarray(image))
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img.float()
        