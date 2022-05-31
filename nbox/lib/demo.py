from nbox import Operator
from nbox.lib.shell import ShellCommand, Python

class YATW(Operator):
  # yet another time waster
  def __init__(self, s=3):
    super().__init__()
    self.s = s

  def forward(self, i = ""):
    import random
    from time import sleep

    y = self.s + random.random()
    print(f"Sleeping (YATW-{i}): {y}")
    sleep(y)


def init_function(s=0.1):
  # this random fuction sleeps for a while and then returns number
  from random import randint
  from time import sleep

  y = randint(4, 8) + s
  print(f"Sleeping (fn): {y}")
  sleep(y)
  return y


class Magic(Operator):
  # named after the Dr.Yellow train that maintains the Shinkansen
  def __init__(self):
    super().__init__()
    self.init = Python(init_function) # waste time and return something
    self.foot = YATW() # Oh, my what will happen here?
    self.cond1 = ShellCommand("echo 'Executing condition {cond}'")
    self.cond2 = ShellCommand("echo 'This is the {cond} option'")
    self.cond3 = ShellCommand("echo '{cond} times the charm :P'")

  def forward(self):
    t1 = self.init()
    self.foot(0)

    if t1 > 6:
      self.cond1(cond = 1)
    elif t1 > 10:
      self.cond2(cond = "second")
    else:
      self.cond3(cond = "Third")
