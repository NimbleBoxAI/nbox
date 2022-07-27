import os
os.environ["NBOX_LOG_LEVEL"] = "warning"

from nbox.relics import Relics
from nbox import Operator

class StoreWriter(Operator):
  def __init__(self):
    super().__init__()
    self.relic = Relics("local:/Users/yashbonde/.nbx/relics")
    
  def forward(self):
    print("writing ...")
    x = "this is a random string"
    self.relic.store.put("/some/object/x", x, ow = True)

class StoreReader(Operator):
  def __init__(self):
    super().__init__()
    self.relic = Relics("local:/Users/yashbonde/.nbx/relics")
    
  def forward(self):
    print("reading ...")
    x = self.relic.store.get("/some/object/x")
    print(x)

class StoreTester(Operator):
  def __init__(self) -> None:
    super().__init__()
    self.write = StoreWriter()
    self.read = StoreReader()

  def forward(self):
    self.write()
    self.read()

if __name__ == "__main__":
  from shutil import rmtree
  rmtree("/Users/yashbonde/.nbx/relics", ignore_errors=True)

  op = StoreTester()
  op()
