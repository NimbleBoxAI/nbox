# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for measuring elapsed time. From TensorBoard (Apache-v2)."""

import contextlib
import logging
import threading
import time

from nbox.utils import logger


def log_latency(region_name_or_function_to_decorate, log_level=None):
  """Log latency in a function or region.
  Three usages are supported. As a decorator:

  ```python
  >>> @log_latency
  ... def function_1():
  ...   pass
  ...
  ```
  
  As a decorator with a custom label for the region:
  ```python
  >>> @log_latency("custom_label")
  ... def function_2():
  ...   pass
  ...
  ```
  
  As a context manager:
  ```python
  >>> def function_3():
  ...   with log_latency("region_within_function"):
  ...     pass
  ...
  ```
  
  - region_name_or_function_to_decorate: Either: a `str`, in which
    case the result of this function may be used as either a
    decorator or a context manager; or a callable, in which case
    the result of this function is a decorated version of that
    callable.
  - log_level: Optional integer logging level constant. Defaults to
    `logging.INFO`.

  Returns:
    A decorated version of the input callable, or a dual
    decorator/context manager with the input region name.
  """

  if log_level is None:
    log_level = logging.DEBUG

  if isinstance(region_name_or_function_to_decorate, str):
    region_name = region_name_or_function_to_decorate
    return _log_latency(region_name, log_level)
  else:
    function_to_decorate = region_name_or_function_to_decorate
    qualname = getattr(function_to_decorate, "__qualname__", None)
    if qualname is None:
      qualname = str(function_to_decorate)
    decorator = _log_latency(qualname, log_level)
    return decorator(function_to_decorate)


class _ThreadLocalStore(threading.local):
  def __init__(self):
    self.nesting_level = 0


_store = _ThreadLocalStore()


@contextlib.contextmanager
def _log_latency(name, log_level):
  if not logger.isEnabledFor(log_level):
    yield
    return

  start_level = _store.nesting_level
  try:
    started = time.time()
    _store.nesting_level = start_level + 1
    thread = threading.current_thread()
    msg = f"ENTER '{name}'"
    if logger.getEffectiveLevel() == 10:
      indent = (" " * 2) * start_level
      msg = f"{thread.name} [{thread.ident}]{indent} {msg}"
    _log(log_level, msg)
    yield
  finally:
    _store.nesting_level = start_level
    elapsed = time.time() - started
    msg = f"LEAVE '{name}' - {elapsed:0.6f}s elapsed"
    if logger.getEffectiveLevel() == 10:
      indent = (" " * 2) * start_level
      msg = f"{thread.name} [{thread.ident}]{indent} {msg}"
    _log(log_level, msg)


def _log(log_level, msg, *args):
  # Forwarding method to ensure that all logging statements
  # originating in this module have the same line number; if the
  # "ENTER" log is on a line with 2-digit number and the "LEAVE" log
  # is on a line with 3-digit number, the logs are misaligned and
  # harder to read.
  logger.log(log_level, msg, *args)

if __name__ == "__main__":
  @log_latency("some fancy function")
  def run(x = 2):
    print("run")
    time.sleep(1)
    return x
  run()

