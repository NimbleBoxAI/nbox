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

import os
import sys
import fire
from json import dumps

from nbox.auth import init_secret
from .jobs import Job, new_model
from .instance import Instance
from .sub_utils import ssh
from .framework.autogen import compile
from .init import nbox_ws_v1
from nbox import utils as U

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

def login():
  init_secret()


class ServeCLINamespace(object):
  new = staticmethod(new_model)

  def __call__(
    self,
    init_folder,
    deployment_id_or_name: str,
    workspace_id: str = None,
    wait_for_deployment: bool = True,
  ):
    from nbox import logger

    folder = os.path.abspath(init_folder)
    sys.path.append(folder)
    logger.info(f"Compiling serving: {folder}")

    # get the items from users code
    from nbx_user import get_op
    try:
      from nbx_user import get_resource
      resource = get_resource()
    except ImportError:
      # old version problems
      resource = None

    from nbox import Operator

    op: Operator = get_op()
    return op.serve(
      init_folder=init_folder,
      deployment_id_or_name = deployment_id_or_name,
      workspace_id = workspace_id,
      resource = resource,
      wait_for_deployment = wait_for_deployment,
    )

def get_dict():
  from .operator import Operator
  NBX = dict(
    login = login                  , # nbox login
    tunnel = ssh.tunnel            , # nbox tunnel
    home = open_home               , # nbox home
    compile = compile              , # nbox compile: internal for autogen code
    get = get                      , # nbox get "/workspace/88fn83/projects"
    #
    build = Instance               , # nbox build
    jobs = Job                     , # nbox jobs
    serve = ServeCLINamespace      , # nbox serve
    unzip = Operator.unzip         , # nbox unzip (convinience)
  )
  return NBX

def main():
  data = get_dict()
  fire.Fire(data)

if __name__ == "__main__":
  main()
