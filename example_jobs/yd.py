#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from time import sleep
from datetime import datetime

from nbox import Operator
from nbox.operators import lib

def init_function(s=0.1):
  # this random fuction sleeps for a while and then returns number
  y = random.randint(4, 8) + s
  print(f"Sleeping (fn): {y}")
  sleep(y)
  return y

class YATW(Operator):
  # yet another time waster
  def __init__(self, s=3):
    super().__init__()
    self.s = s

  def forward(self):
    y = self.s + random.random()
    print(f"Sleeping (YATW): {y}")
    sleep(y)

class DoctorYellow(Operator):
  # named after the Dr.Yellow train that maintains the Shinkansen
  def __init__(self):
    super().__init__()
    self.init = lib.Python(init_function) # waste time and return something
    self.step2 = lib.ShellCommand("echo 'init command slept for {number} seconds'") # fast
    self.foot = YATW()

    self.cond1 = lib.ShellCommand('echo "cond1"', 'echo "current time {time}"')
    self.cond2 = lib.ShellCommand('echo "Cond2: This should never happen"')
    self.cond3 = YATW(4)

  def forward(self):
    print("-" * 70)
    t1 = self.init(); print("-" * 70)
    self.step2(number = t1); print("-" * 70)
    self.foot(); print("-" * 70)

    if t1 > 6:
      self.cond1(time = datetime.now().isoformat())
    elif t1 > 10:
      self.cond2()
    else:
      self.cond3()

op = DoctorYellow()
print(op)

from pprint import pprint
pprint(op.deploy(None))
