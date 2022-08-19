"""
This is CLI for ``nbox``, it is meant to be as simple to use as possible. We using ``python-fire`` for building
our CLI, which means you can access underlying system with same commands as you use with python. For maximum
convinience we recommend adding an alias for ``nbx`` as follows:

.. code-block:: bash
  
  [zshrc]
  echo "\\nalias nbx='python3 -m nbox'\\n" >> ~/.zshrc
  source ~/.zshrc

  [bashrc]
  echo "\\nalias nbx='python3 -m nbox'\\n" >> ~/.bashrc
  source ~/.bashrc


SSH into Instances
------------------

You can SSH into your instance with the ``nbx tunnel`` command. This command will create a tunnel
to your instance and start an SSH session.

.. code-block:: bash

  nbx tunnel 8000 -i "instance-name"

Documentation
-------------
"""

import os
import sys
import fire
from json import dumps

import nbox.utils as U
from nbox.jobs import Job, Serve
from nbox.init import nbox_ws_v1
from nbox.auth import init_secret
from nbox.instance import Instance
from nbox.sub_utils.ssh import tunnel
from nbox.relics import RelicsNBX
from nbox.version import __version__ as V

logger = U.get_logger()

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
  """Re authenticate ``nbox``. NOTE: Will remove all added keys to ``secrets``"""
  fp = U.join(U.env.NBOX_HOME_DIR(), "secrets.json")
  os.remove(fp)
  init_secret()

def version():
  logger.info("NimbleBox.ai Client Library")
  logger.info(f"    nbox version: {V}")


def main():
  fire.Fire({
    "build"   : Instance,
    "get"     : get,
    "jobs"    : Job,
    "login"   : login,
    "open"    : open_home,
    "relics"  : RelicsNBX,
    "serve"   : Serve,
    "tunnel"  : tunnel,
    "version" : version,
  })

if __name__ == "__main__":
  main()
