"""
This is CLI for ``nbox``, it is meant to be as simple to use as possible.

..code-block::

  nbx
  ├── tunnel
  ├── home
  ├── compile
  ├── get
  ├── jobs
  │   [staticmethods]
  │   ├── new
  │   ├── status
  │   [initialisation]
  │       ├── id
  │       └── workspace_id
  │   ├── change_schedule
  │   ├──logs
  │   ├──delete
  │   ├──refresh
  │   ├──trigger
  │   ├──pause
  │   └──resume
  └── build (Instance)
      [staticmethods]
      ├── new
      ├── status
      [initialisation]
          ├── i
          ├── workspace_id
          └── cs_endpoint
      ├── refresh
      ├── start
      ├── stop
      ├── delete
      ├── run_py
      └── __call__

"""

import sys
import fire
from json import dumps

from .jobs import Job
from .instance import Instance
from .sub_utils import ssh
from .framework.autogen import compile
from .init import nbox_ws_v1


def open_home():
  """Open current NBX platform"""
  from .auth import secret
  import webbrowser
  webbrowser.open(secret.get("nbx_url"))


def get(api_end: str, **kwargs):
  """Get any command 

  Args:
    api_end (str): something like '/v1/api'
  """
  api_end = api_end.strip("/")
  if nbox_ws_v1 == None:
    raise RuntimeError("Not connected to NimbleBox.ai webserver")
  out = nbox_ws_v1
  for k in api_end.split("/"):
    out = getattr(out, k)
  res = out(_method = "get", **kwargs)
  sys.stdout.write(dumps(res))
  sys.stdout.flush()

NBX = dict(
  tunnel = ssh.tunnel            , # nbox tunnel
  home = open_home               , # nbox home
  compile = compile              , # nbox compile: internal for autogen code
  get = get                      , # nbox get "/workspace/88fn83/projects"
  #
  build = Instance               , # nbox build
  jobs = Job                     , # nbox jobs
)

if __name__ == "__main__":
  fire.Fire(NBX)
