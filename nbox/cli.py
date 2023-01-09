"""
This is CLI for `nbox`, it is meant to be as simple to use as possible. We using `python-fire` for building
our CLI, which means you can access underlying system with same commands as you use with python.

# SSH into Instances

You can SSH into your instance with the `nbx tunnel` command. This command will create a tunnel
to your instance and start an SSH session.

```bash
nbx tunnel 8000 -i "instance-name"
```
"""

import os
import fire
import jinja2
import tempfile
import webbrowser
from json import dumps
from typing import Dict, Any

import nbox.utils as U
from nbox.jobs import Job, Serve
from nbox.init import nbox_ws_v1
from nbox.auth import init_secret, ConfigString, secret
from nbox.instance import Instance
from nbox.sub_utils.ssh import tunnel
from nbox.relics import RelicsNBX
from nbox.lmao import LmaoCLI
from nbox.version import __version__ as V
from nbox.nbxlib.fire import NBXFire

logger = U.get_logger()


class Config(object):
  def update(self, workspace_id: str = ""):
    """Set global config for `nbox`"""
    secret = init_secret()
    if workspace_id:
      secret.put(ConfigString.workspace_id, workspace_id, True)
      logger.info(f"Global Workspace ID set to {workspace_id}")

      data = secret.get(ConfigString.cache)
      redo = not data or (workspace_id not in data)

      if redo:
        workspaces = nbox_ws_v1.workspace()
        workspace_details = list(filter(lambda x: x["workspace_id"] == workspace_id, workspaces))
        if len(workspace_details) == 0:
          logger.error(f"Could not find the workspace ID: {workspace_id}. Please check the workspace ID and try again.")
          raise Exception("Invalid workspace ID")
        workspace_details = workspace_details[0]
        workspace_name = workspace_details["workspace_name"]
        secret.secrets.get(ConfigString.cache).update({workspace_id: workspace_details})
      else:
        data = data[workspace_id]
        workspace_name = data["workspace_name"]

      secret.put(ConfigString.workspace_name, workspace_name, True)
      logger.info(f"Global Workspace: {workspace_name}")

  def show(self):
    """Pretty print global config for `nbox`"""
    workspace_id = secret.get(ConfigString.workspace_id)
    workspace_name = secret.get(ConfigString.workspace_name)
    logger.info(
      "\nnbox config:\n" \
      f"  workspace_name: {workspace_name}\n" \
      f"    workspace_id: {workspace_id}\n" \
      f"    nbox version: {V}\n" \
      f"             URL: {secret.get('nbx_url')}"
    )

def open_home():
  """Open current NBX platform"""
  from .auth import secret
  import webbrowser
  webbrowser.open(secret.get("nbx_url"))


def get(api_end: str, no_pp: bool = False, **kwargs):
  """GET any API.

  Args:
    api_end (str): something like '/v1/api'
    no_pp (bool, optional): Pretty print. Defaults to False.
  """
  api_end = api_end.strip("/")
  if nbox_ws_v1 == None:
    raise RuntimeError("Not connected to NimbleBox.ai webserver")
  out = nbox_ws_v1
  for k in api_end.split("/"):
    out = getattr(out, k)
  res = out(_method = "get", **kwargs)
  if not no_pp:
    logger.info(dumps(res, indent=2))
  else:
    return res

def login():
  """Re authenticate `nbox`. NOTE: Will remove all added keys to `secrets`"""
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


class NBXWS_CLI(object):
  def get(self, api: str, H: Dict[str, str] = {}):
    """make a GET call to any `api`, it can either contain '/' or '.'"""
    if nbox_ws_v1 == None:
      raise RuntimeError("Not connected to NimbleBox.ai webserver")
    api = api.strip("/")
    splitter = '.' if '.' in api else '/'
    nbox_ws_v1._session.headers.update(H)
    out = nbox_ws_v1
    for k in api.split(splitter):
      out = getattr(out, k)
    res = out(_method = "get")
    return res

  def post(self, api: str, d: Dict[str, Any] = {}, H: Dict[str, str] = {}):
    """make a POST call to any `api`, it can either contain '/' or '.'"""
    if nbox_ws_v1 == None:
      raise RuntimeError("Not connected to NimbleBox.ai webserver")
    api = api.strip("/")
    splitter = '.' if '.' in api else '/'
    nbox_ws_v1._session.headers.update(H)
    out = nbox_ws_v1
    for k in api.split(splitter):
      out = getattr(out, k)
    res = out(_method = "post", data = d)
    return res

  # def rapidoc(self):
  #   openapi = secret.get("openapi_spec", None)
  #   if openapi == None:
  #     raise RuntimeError("Not connected to NimbleBox.ai webserver")
  #   fp = U.join(U.folder(__file__), "assets", "rapidoc.html")
  #   with open(fp, "r") as f:
  #     temp = jinja2.Template(f.read())
  #   with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".html") as f:
  #     f.write(temp.render(openapi = dumps(openapi)))
  #     print(f.name)
  #     webbrowser.open(f"file://{f.name}")


def main():
  component = {
    "build"   : Instance,
    "config"  : Config,
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
    "ws"      : NBXWS_CLI,
  }

  fire.Fire(component)
  # NBXFire(component)

if __name__ == "__main__":
  main()
