import torch
import unittest

from nbox import Model
from nbox.framework.on_ml import ModelOutput


class SampleModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.sample = {
      "x": torch.randn(2, 8),
      "y": torch.randn(2, 8),
    }
    self.a = torch.nn.Linear(8, 16)
    self.b = torch.nn.Linear(8, 16)

  def forward(self, x, y):
    return self.a(x) + self.b(y)


class Test_Model(unittest.TestCase):
  def test_model_wo_input_processing(self):
    _m = SampleModel()
    m: ModelOutput = Model(_m, None) # define the model with DAG and no input processing
    out = m(_m.sample) # pass the input directly, and it will work because it is a dict
    self.assertEqual(out.outputs.shape, torch.Size([2, 16]))

  def test_model_w_input_processing_A(self):
    _m = SampleModel()
    def process_input(inputs):
      return {
        "x": torch.from_numpy(inputs["x"]).float(),
        "y": torch.from_numpy(inputs["y"]).float(),
      }
    
    m = Model(_m, process_input)
    out = m({x:y.numpy() for x,y in _m.sample.items()})
    self.assertEqual(out.outputs.shape, torch.Size([2, 16]))

  def test_model_w_incorrect_processing_A(self):
    _m = SampleModel()
    m = Model(_m, None)

    with self.assertRaises(TypeError):
      out = m({x:y.tolist() for x,y in _m.sample.items()})
      self.assertEqual(out.outputs.shape, torch.Size([2, 16]))

