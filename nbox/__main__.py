"""
nboxCLI
=======

This is CLI for ``nbox``, it is meant to be as simple to use as possible.
The commands are broken down according to the products they are related to.

.. code-block::

  nbox instance [nbox.Instance] **init_kwargs [actions] **acton_kwargs
  nbox jobs [new/deploy/open]
"""

import sys
import fire
from json import dumps

from . import cli as n # nbox-cli
from .jobs import Job
from .instance import Instance
from .sub_utils import ssh
from .framework.autogen import compile
from .init import nbox_ws_v1

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
  build = Instance               , # nbox build
  jobs = Job                     , # nbox jobs
  tunnel = ssh.tunnel            , # nbox tunnel
  home = n.open_home             , # nbox home
  compile = compile              , # nbox compile: internal for autogen code
  get = get                      , # nbox get "/workspace/88fn83/projects"
)

if __name__ == "__main__":
  fire.Fire(NBX)
