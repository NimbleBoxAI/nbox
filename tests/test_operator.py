import unittest

from nbox.operators.operator import *

# operators/

class AddTwoNos(Operator):
  def __init__(self):
    super().__init__()
  
  def forward(self, a, b):
    return a + b

class AddFourNos(Operator):
  def __init__(self):
    super().__init__()
    self.op = AddTwoNos()

  def forward(self, a, b, c, d):
    return self.op(a, b) + self.op(c, d)

class AddSixNos(Operator):
  def __init__(self):
    super().__init__()
    self.op1 = AddFourNos()
    self.op2 = AddTwoNos()

  def forward(self, a, b, c, d, e, f):
    return self.op1(a, b, c, d) + self.op2(e, f)

# /operators

class OperatorTest(unittest.TestCase):
  def test_one_operator(self):
    op = AddTwoNos()
    self.assertEqual(op(1, 2), 3)

  def test_chained(self):
    op = AddSixNos()
    self.assertEqual(op(1, 2, 3, 4, 5, 6), 21)
