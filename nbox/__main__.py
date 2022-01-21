"""
nboxCLI
=======

This is CLI for ``nbox``, it is meant to be as simple to use as possible.
The commands are broken down according to the products they are related to.

.. code-block:: python

    {
      "deploy": n.deploy,           # nbox deploy
      "jobs": {
        "instance": {
          "start": j.start,         # nbox jobs instance start
          "stop": j.stop,           # nbox jobs instance stop
        },
        "init": j.init,             # nbox jobs init
        "deploy": j.deploy,         # nbox jobs deploy
        "status": j.status,         # nbox jobs status
      },
      "status": n.status,           # nbox status
      "tunnel": n.tunnel,           # nbox tunnel
    }
"""

import fire
<<<<<<< HEAD
from . import cli as n # nbox-cli
from .jobs import cli as j # jobx-cli

if __name__ == "__main__":
  fire.Fire({
    "deploy": n.deploy,           # nbox deploy
    "jobs": {
      "instance": {
        "start": j.start,         # nbox jobs instance start NAME (...)
        "stop": j.stop,           # nbox jobs instance stop NAME (...)
      },
      "init": j.init,             # nbox jobs init PROJECT_NAME
      "deploy": j.deploy,         # nbox jobs deploy FOLDER
      "status": j.status,         # nbox jobs status ID_OR_NAME
    },
    "status": n.status,           # nbox status
    "tunnel": n.tunnel,
  })
=======
from .cli import deploy
from .jobs import print_status
from .jobs.cli import jobs_cli

if __name__ == "__main__":
    fire.Fire({
        "deploy": deploy,
        "jobs": jobs_cli,
        "status": lambda loc = None: print_status(f"https://{'' if not loc else loc+'.'}nimblebox.ai")
    })
>>>>>>> master
