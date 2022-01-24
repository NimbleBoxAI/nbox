import random
from time import sleep

from nbox import Operator
from nbox.operators import lib

def init_function(s=0.1):
  y = random.randint(5, 8) + s
  sleep(y)
  return y

class DoctorYellow(Operator):
  # named after the yellow coloured train that maintains the Shinkansen
  def __init__(self):
    super().__init__()
    self.init = lib.Python(init_function)

  def forward(self):
    t1 = self.init()

op = DoctorYellow()
