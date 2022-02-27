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
from .sub_utils import ssh

if __name__ == "__main__":
  fire.Fire({
    "instance": Instance,         # nbox jobs instance
    "jobs": {
      "new": ij.new_job,          # nbox jobs new PROJECT_NAME
      "deploy": ij.deploy,        # nbox jobs deploy FOLDER
      "open": ij.open_jobs,       # nbox jobs open
      # "trigger":                # trigger a job run from CLI
    },
    "status": Instance.print_status, # nbox status
    "tunnel": ssh.tunnel,
    "open": n.open_home,
  })
