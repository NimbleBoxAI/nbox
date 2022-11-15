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

logger = U.get_logger()

def More(contents: str, out):
  """Run a user specified pager or fall back to the internal pager.

  Args:
    contents: The entire contents of the text lines to page.
    out: The output stream.
    prompt: The page break prompt.
    check_pager: Checks the PAGER env var and uses it if True.
  """
  import signal, subprocess

  pager = encoding.GetEncodedValue(os.environ, 'PAGER', None)
  if pager == '-':
    # Use the fallback Pager.
    pager = None
  elif not pager:
    # Search for a pager that handles ANSI escapes.
    for command in ('less', 'pager'):
      if files.FindExecutableOnPath(command):
        pager = command
        break
  if pager:
    # If the pager is less(1) then instruct it to display raw ANSI escape
    # sequences to enable colors and font embellishments.
    less_orig = encoding.GetEncodedValue(os.environ, 'LESS', None)
    less = '-R' + (less_orig or '')
    encoding.SetEncodedValue(os.environ, 'LESS', less)
    # Ignore SIGINT while the pager is running.
    # We don't want to terminate the parent while the child is still alive.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    p = subprocess.Popen(pager, stdin=subprocess.PIPE, shell=True)
    enc = console_attr.GetConsoleAttr().GetEncoding()
    p.communicate(input=contents.encode(enc))
    p.wait()
    # Start using default signal handling for SIGINT again.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if less_orig is None:
      encoding.SetEncodedValue(os.environ, 'LESS', None)
    return
  else:
    out.write(contents)


class TC:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class NBXFire:
  """This is the CLI function for bespoke designed for nbox. Names after the legendary `python-fire` command which served
  us well for many years before we ended up here."""
  def __init__(self, component):
    # print(os.path.basename(sys.argv[0]))
    args = sys.argv[1:]
    service_level = args[0]
    if service_level not in component or service_level == '--help':
      lines = [
        f"{TC.BOLD}{TC.FAIL}ERROR:{TC.ENDC} Could not find command: '{args[0]}'. Available commands are:\n"
      ] + [f"  {x}" for x in tuple(component.keys())]

    text = '\n'.join(lines) + '\n\n'
    More(text, out=sys.stderr)
    # print(text)

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
        secret.secrets.get(ConfigString.cache.value).update({workspace_id: workspace_details})
      else:
        data = data[workspace_id]
        workspace_name = data["workspace_name"]

      secret.put(ConfigString.workspace_name, workspace_name, True)
      logger.info(f"Global Workspace: {workspace_name}")

  def show(self):
    """Pretty print global config for `nbox`"""
    workspace_id = secret.get(ConfigString.workspace_id.value)
    workspace_name = secret.get(ConfigString.workspace_name.value)
    logger.info(
      "\nnbox config:\n" \
      f"  workspace_id: {workspace_id}\n" \
      f"  workspace_name: {workspace_name}\n" \
      f"  nbox version: {V}" \
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
  })

  fire.Fire(component)
  # NBXFire(component)

if __name__ == "__main__":
  main()
