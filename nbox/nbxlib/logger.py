"""
Contains our custom logger and formatters so you can log things like a pro.
It can generate both inline logs as well as JSON logs (easier to consume by other tools).
Json logs go really well with our custom logger for jobs and deploy where you can
logs arbitrary objects and have them nicely formatted in the console.

Parts of this code are taken from the awesome `pyjsonlogger` library
([Pypi](https://pypi.org/project/jsonformatter/))
"""
# BSD 2-Clause License
# 
# Copyright (c) 2019, MyColorfulDays
# Copyright (c) 2023, NimbleBox.ai
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import json
import logging
import traceback
from inspect import istraceback
from datetime import date, datetime, time, timezone


def get_logger(env):
  logger = logging.getLogger("nbox")
  lvl = env.NBOX_LOG_LEVEL("info").upper()
  logger.setLevel(getattr(logging, lvl))

  if env.NBOX_JSON_LOG(False):
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(JsonFormatter())
    logger.addHandler(logHandler)
  else:
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(InlineFormatter(
      '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
      datefmt = "%Y-%m-%dT%H:%M:%S%z"
    ))
    logger.addHandler(logHandler)

  return logger


##############

class _LogObject:
  def __init__(self, log, msg):
    self.log = log
    self.msg = msg

def lo(msg, /, *args, **kwargs):
  if args:
    msg += " " + " ".join([str(a) for a in args])
  return _LogObject(kwargs, msg)

##############

class InlineFormatter(logging.Formatter):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def format(self, record):
    if type(record.msg) == _LogObject:
      _l: _LogObject = record.msg
      if _l.log:
        record.msg = f"{_l.msg}\n" + json_dumps_pretty(_l.log)
      else:
        record.msg = _l.msg
    return super().format(record)

class JsonFormatter(logging.Formatter):
  def format(self, record):
    """Formats a log record and serializes to json"""

    log_record = {
      "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
      "levelname": record.levelname,
      "logtype": "nbox_jlv1",
      "log": {
        "code": {
          "file": record.filename,
          "line": record.lineno,
        },
      }
    }

    if isinstance(record.msg, dict):
      log_record["message"] = "click to see data"
      log_record["log"]["data"] = record.msg
    elif isinstance(record.msg, _LogObject):
      _lo: _LogObject = record.msg
      log_record["message"] = _lo.msg
      if _lo.log:
        log_record["log"]["data"] = _lo.log
    elif record.msg:
      lines = record.getMessage().splitlines()
      if len(lines) == 1:
        # you only have one line, don't bother to create lot of data
        log_record["message"] = lines[0]
      else:
        log_record["message"] = lines[0][:10] + "..."
        log_record["log"]["lines"] = lines

    json_str = json_dumps_tight(log_record)
    return json_str

##############

class _JsonEncoder(json.JSONEncoder):
  """
  A custom encoder extending the default JSONEncoder
  """
  def default(self, obj):
    if isinstance(obj, (date, datetime, time)):
      return self.format_datetime_obj(obj)
    elif istraceback(obj):
      return ''.join(traceback.format_tb(obj)).strip()
    elif type(obj) == Exception \
        or isinstance(obj, Exception) \
        or type(obj) == type:
      return str(obj)
    try:
      return super(_JsonEncoder, self).default(obj)
    except TypeError:
      try:
        return str(obj)
      except Exception:
        return None

  def format_datetime_obj(self, obj):
    return obj.isoformat()


json_dumps_tight = lambda log_record: json.dumps(
  log_record,
  cls = _JsonEncoder,
  ensure_ascii = True,
  separators = (',', ':'),
)

json_dumps_pretty = lambda log_record: json.dumps(
  log_record,
  cls = _JsonEncoder,
  ensure_ascii = True,
  indent = 2,
  separators = (',', ': '),
)