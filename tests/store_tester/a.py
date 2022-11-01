import os
os.environ["NBOX_LOG_LEVEL"] = "warning"

from nbox.relics import LocalStore
from nbox import Operator

class StoreWriter(Operator):
  def __init__(self):
    super().__init__()
    self.relic = LocalStore()
    
  def forward(self):
    print("writing ...")
    x = "this is a random string"
    self.relic.put("/some/object/x", x, ow = True)

class StoreReader(Operator):
  def __init__(self):
    super().__init__()
    self.relic = LocalStore()
    
  def forward(self):
    print("reading ...")
    x = self.relic.get("/some/object/x")
    print(x)

class StoreTester(Operator):
  def __init__(self) -> None:
    super().__init__()
    self.write = StoreWriter()
    self.read = StoreReader()

  def forward(self):
    self.write.forward()
    self.read.forward()

if __name__ == "__main__":
  op = StoreTester()
  op()
