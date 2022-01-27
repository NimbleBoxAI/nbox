#!/usr/bin/env python

import os
import sys
from nbox.utils import folder, join
from glob import glob

from common import ModuleNotFound

all_tests = [x for x in glob(join(folder(__file__), "test_*")) if os.path.isdir(x)]

for a in all_tests:
  print("Testing in folder:", a)
  sys.path.insert(0, a)
  from init import main
  try:
    main()
  except ModuleNotFound as e:
    print(f"Module: {e} not found, skipping")
    sys.path.pop(0)
    continue

  import subprocess
  subprocess.call(["python", "-m", "pytest", f"{a}/tests.py", "-v", "-s"])
  sys.path.pop(0)
