from nbox.jobs.operator import *

class AddTwoNos(Operator):
  def __init__(self):
    super().__init__()
  
  def forward(self, a, b):
    return a + b

op = AddTwoNos()
# print(op._operators)
# print(op.__dict__)
print(op)
print(tuple(op.named_operators()))
# print(op(1, 2))
print("="  * 20)

class AddFourNos(Operator):
  def __init__(self):
    super().__init__()
    self.op = AddTwoNos()

  def forward(self, a, b, c, d):
    return self.op(a, b) + self.op(c, d)

op = AddFourNos()
# print(op._operators)
# print(op.__dict__)
print(op)
print("---------")
for name, op in op.named_operators():
  print("  ", name, op)
print("="  * 20)

class AddSixNos(Operator):
  def __init__(self):
    super().__init__()
    self.op1 = AddFourNos()
    self.op2 = AddTwoNos()

  def forward(self, a, b, c, d, e, f):
    return self.op1(a, b, c, d) + self.op2(e, f)

op = AddSixNos()
# print(op._operators)
# print(op.__dict__)
print(op)
print("---------")
for idx, (name, _op) in enumerate(op.named_operators()):
  # if not idx: continue
  print("  ", name, "::")

# print(op)
# print("---------")
# for idx, (name, _op) in enumerate(op.named_operators()):
#   # if not idx: continue
#   print("  ", name, "::")

print(op(1, 2, 3, 4, 5, 6))
print("="  * 20)



from torch.nn import Module, Linear
