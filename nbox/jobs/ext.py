import time
import inspect
from collections import Counter

import logging
logger = logging.getLogger("jobs.utils")

from .utils import Subway, TIMEOUT_CALLS


class NBXServerFunctions:

  def update(self):
    pass


class ShellFunctions:
  def cp(self):
    pass

  def mv(self):
    pass

  def rm(self):
    pass

  def ls(self):
    pass

  def mkdir(self):
    pass

  def cat(self):
    pass

  def head(self):
    pass

  def tail(self):
    pass


# now we need to check if any of the classes have an overlapping functions
__all_methods = []
for x in [NBXServerFunctions, ShellFunctions]:
  __all_methods.extend([
    x[0] for x in inspect.getmembers(x, predicate=inspect.isfunction) if not x[0].startswith("_")
  ])

err = False
for x,y in Counter(__all_methods).items():
  if y > 1:
    logger.error(f"{x} is defined in multiple classes")
    err = True

if err:
  raise ValueError("overlapping methods in classes, please check logs")

