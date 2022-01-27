import torch
import unittest
import numpy as np

from nbox import Model


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
    m = Model(
      _m,
      {
        "x": None,
        "y": None,
      }
    )

    x = _m.sample
    out = m(x)
    self.assertEqual(out.shape, torch.Size([2, 16]))

  def test_model_w_input_processing_A(self):
    _m = SampleModel()
    m = Model(
      _m,
      {
        "x": torch.from_numpy,
        "y": torch.from_numpy,
      }
    )

    x = {x:y.numpy().astype(np.float32) for x,y in _m.sample.items()}
    out = m(x)
    self.assertEqual(out.shape, torch.Size([2, 16]))

  def test_model_w_input_processing_B(self):
    _m = SampleModel()
    m = Model(
      _m,
      {
        "x": lambda x: torch.tensor(x, dtype=torch.float32),
        "y": lambda x: torch.tensor(x, dtype=torch.float32),
      }
    )

    x = {x:y.tolist() for x,y in _m.sample.items()}
    out = m(x)
    self.assertEqual(out.shape, torch.Size([2, 16]))

  @unittest.expectedFailure
  def test_model_w_incorrect_processing_A(self):
    _m = SampleModel()
    m = Model(
      _m,
      {
        "x": torch.from_numpy,
        "y": torch.from_numpy,
      }
    )

    x = {x:y.tolist() for x,y in _m.sample.items()}
    out = m(x)
    self.assertEqual(out.shape, torch.Size([2, 16]))

