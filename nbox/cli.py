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
from nbox.auth import init_secret, ConfigString
from nbox.instance import Instance
from nbox.sub_utils.ssh import tunnel
from nbox.relics import RelicsNBX
from nbox.lmao import LmaoCLI
from nbox.version import __version__ as V

logger = U.get_logger()

def global_config(workspace_id: str = ""):
  """Set global config for ``nbox``"""
  secret = init_secret()
  if workspace_id:
    secret.put(ConfigString.workspace_id.value, workspace_id, True)
    logger.info(f"Global Workspace ID set to {workspace_id}")


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


def why():
  print("\nWhy we build NimbleBox?\n")
  print("  * Artificial intelligence will be the most important technology of the 21st century.")
  print("  * Every major piece of software ever written will need to be upgraded and rewritten.")
  print("  * Energy spent per code token will increase exponentially to handle the bandwidth of AI.")
  print("  * AI is still software and software engineering is hard.")
  print("  * Nimblebox is a general purpose tool to build and manage such operations.")
  print("\nIf you like what we are building, come work with us.\n\nWith Love,\nNimbleBox.ai\n")


def main():
  fire.Fire({
    "build"   : Instance,
    "config"  : global_config,
    "get"     : get,
    "jobs"    : Job,
    "lmao"    : LmaoCLI,
    "login"   : login,
    "open"    : open_home,
    "relics"  : RelicsNBX,
    "serve"   : Serve,
    "tunnel"  : tunnel,
    "version" : version,
    "why"     : why,
  })

if __name__ == "__main__":
  main()
