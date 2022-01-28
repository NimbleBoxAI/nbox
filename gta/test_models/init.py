import os
import sys
from nbox.utils import folder, join, _isthere

# sys.path.append(join(folder(folder(__file__))))

def main():
  mapping = {
    # just the forward pass
    "NBOX_TEST_SKLEARN_FORWARD": {
      "condition": int(_isthere('sklearn')),
      "functions": []
    },
    "NBOX_TEST_TORCH_FORWARD": {
      "condition": int(_isthere('torch')),
      "functions": []
    },
    "NBOX_TEST_ONNXRT_FORWARD": {
      "condition": int(_isthere('onnxruntime')),
      "functions": []
    },

    "NBOX_TEST_TORCH_2_ONNX_DEPLOY": {
      "condition": int(_isthere('torch') and _isthere('onnx')),
      "functions": []
    },
    "NBOX_TEST_TORCH_2_TORCHSCRIPT": {
      "condition": int(_isthere('torch')),
      "functions": []
    },
    "NBOX_TEST_SKLEARN_2_ONNX_DEPLOY": {
      "condition": int(_isthere('sklearn') and _isthere('skl2onnx')),
      "functions": []
    },
    "NBOX_TEST_SKLEARN_2_PICKLE": {
      "condition": int(_isthere('sklearn')),
      "functions": []
    },
  }