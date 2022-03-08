"""
nboxCLI
=======

This is CLI for ``nbox``, it is meant to be as simple to use as possible.
The commands are broken down according to the products they are related to.

.. code-block::

  nbox instance [nbox.Instance] **init_kwargs [actions] **acton_kwargs
  nbox jobs [new/deploy/open]
"""

import fire
from . import cli as n # nbox-cli
from .jobs import Job
from .instance import Instance
from .sub_utils import ssh

NBX = dict(
  instance = Instance            , # nbox instance
  jobs = Job                     , # nbox jobs
  tunnel = ssh.tunnel            , # nbox tunnel
  home = n.open_home             , # nbox home
)

if __name__ == "__main__":
  fire.Fire(NBX)
