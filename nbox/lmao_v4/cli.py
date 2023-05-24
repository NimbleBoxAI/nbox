"""
NimbleBox LMAO is our general purpose observability tool for any kind of computation you might have.
"""

from typing import Optional
from dateparser import parse as parse_date
from google.protobuf.timestamp_pb2 import Timestamp

from nbox.utils import logger
from nbox.auth import secret, AuthConfig
from nbox import messages as mpb

# all the sublime -> hyperloop stuff
from nbox.lmao_v4.proto.project_pb2 import ListProjectsRequest, ListProjectsResponse
from nbox.lmao_v4.proto.logs_pb2 import TrackerLogRequest, TrackerLogResponse
from nbox.lmao_v4 import common

class Project:
  def list(self):
    lmao_stub = common.get_lmao_stub()
    out: ListProjectsResponse = lmao_stub.ListProjects(ListProjectsRequest())
    return out


def serving_logs(
  project_id: str,
  tracker_id: str,
  key: str = "",
  after: Optional[str] = "",
  before: Optional[str] = "",
  limit: int = 100,
  f: str = "",
):
  """
  Get your logs from a serving by key and time range.

  ```bash
  nbx lmao serving logs '229dj92' '0283-nice-summer' key 'last 12 hours'
  nbx lmao serving logs '229dj92' '0283-nice-summer' '2021-01-01' '2021-01-02'
  nbx lmao serving logs '229dj92' '0283-nice-summer' key '2021-01-01 12:00:00' '2021-01-02 12:00:00'
  nbx lmao serving logs '229dj92' '0283-nice-summer' --limit 1000
  nbx lmao serving logs '229dj92' '0283-nice-summer' key --limit 1000 --f logs.json
  ```

  Args:
    project_id: ID of the project to which the serving belongs.
    serving_id: ID of the serving to which the log belongs.
    key: Key of the log to get.
    after: Return only logs after this timestamp.
    before: Return only logs before this timestamp.
    limit: Maximum number of logs to return.
    f: File to write logs to. If not provided, logs will be printed to stdout.
  """
  stub = common.get_lmao_stub()
  req = TrackerLogRequest(
    project_id = project_id,
    tracker_id = tracker_id,
    keys = [key,]
  )

  # date things
  if after:
    after_parsed = parse_date(after, settings={'TIMEZONE': 'UTC'})
    if after_parsed is None:
      raise ValueError(f"Invalid date format for `after` argument: {after}")
    t = Timestamp()
    t.FromDatetime(after_parsed)
    req.after.CopyFrom(t)
    logger.info(f"Will get logs from after: {after_parsed}")
  if before:
    before_parsed = parse_date(before, settings={'TIMEZONE': 'UTC'})
    if before_parsed is None:
      raise ValueError(f"Invalid date format for `before` argument: {before}")
    t = Timestamp()
    t.FromDatetime(before_parsed)
    req.before.CopyFrom(t)
    logger.info(f"Will get logs from before: {before_parsed}")
  if after and before:
    if after_parsed > before_parsed:
      raise ValueError(f"`after` date cannot be greater than `before` date")

  # limit things
  if limit < 0:
    raise ValueError("`limit` cannot be negative")
  req.limit = int(limit)

  logs: TrackerLogResponse = stub.get_serving_log(req)
  logs_json = mpb.message_to_json(
    message = logs,
    including_default_value_fields = False,
    sort_keys = True,
    use_integers_for_enums = False
  )
  if not f:
    print(logs_json)
    return
  logger.info(f"Writing logs to {f}")
  with open(f, "w") as _f:
    _f.write(logs_json)

# add all the CLIs at the bottom here:
LmaoCLI = {
  "project": Project, 
  "serving": serving_logs,
}
