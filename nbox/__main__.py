"""
nboxCLI
=======

This is CLI for ``nbox``, it is meant to be as simple to use as possible.
The commands are broken down according to the products they are related to.

.. code-block::

  nbox deploy [OPTIONS]
  nbox instance [OPTIONS-1] [create/start/stop] [OPTIONS-2]
  nbox jobs [init/]
"""

import fire
from . import cli as n # nbox-cli
from .jobs import cli as ij # jobx-cli
from .jobs import Instance

if __name__ == "__main__":
  fire.Fire({
    "deploy": n.deploy,           # nbox deploy
    "instance": Instance,         # nbox jobs instance
    "jobs": {
      "new": ij.new_job,          # nbox jobs init PROJECT_NAME
      "deploy": ij.deploy,        # nbox jobs deploy FOLDER
    },
    "status": Instance.print_status, # nbox status
    "tunnel": n.tunnel,
    "open": n.open_home,
  })
