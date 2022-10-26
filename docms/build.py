import os
os.environ["NBOX_LOG_LEVEL"] = "error"
os.environ["NBOX_NO_AUTH"] = "1"
os.environ["NBOX_NO_LOAD_GRPC"] = "1"
os.environ["NBOX_NO_LOAD_WS"] = "1"
os.environ["NBOX_NO_CHECK_VERSION"] = "1"

import nbox
import nbox.utils as U

import re
from fire import Fire
from typing import List
from functools import partial

STATE_FILE = "nbx_docs.state.json"

def create_blanks(code: str, src: str, gen: str, ignore_pat: List[str] = [], fresh: bool = False):
  """This function will go over the nbox folder and create blank files associated for each. You can get a fresh start by
  passing --fresh flag."""
  # Here's how the entire thing works, there are 2 input files "code" (.py) and "src" (.md) that are used to generate the
  # final "gen" (.md) file. Now when generating a fresh copy we don't need to run over all the computations again, so we
  # will store everything in a state file (nbx_docs.state.json). This state will contain a "hashes" key which will be a
  # dict of "module_name" and "code" and "src" SHA256 hashes. If the hashes are the same and "gen" exists, we will skip
  # updating it.
  if fresh:
    y = input("Are you sure you want to delete all the existing files? (y/n): ")
    if y.lower() != "y":
      print("Aborting...")
      return
    os.system(f"rm {src}code/*.md")

  print(f"f('{code}', '{src}') -> '{gen}'")
  nbox_files = U.get_files_in_folder(code, ".py", False)
  src_files = U.get_files_in_folder(U.join(src, "code"), ".md", False)
  gen_files = U.get_files_in_folder(U.join(gen, "code"), ".md", False)
  print(f"Files in nbox/ folder: {len(nbox_files)} ")
  print(f"Files in src/ folder: {len(src_files)} ")
  print(f"Files in gen/ folder: {len(gen_files)} ")

  # load the state and move forward from here
  if not os.path.exists(STATE_FILE):
    state = {"hashes": {}}
  else:
    state = U.from_json(STATE_FILE)

  # get the hashes of all the files and match to the state
  mod_to_do = set()
  mod_to_file = {}
  ig_pats = []
  for p in ignore_pat:
    pat = ""
    if not p.startswith("^"):
      pat += "^.*"
    pat += p
    if not p.endswith("$"):
      pat += ".*$"
    # print(pat)
    ig_pats.append(re.compile(pat))
  print(ig_pats)

  for nbox_file in nbox_files:
    with open(nbox_file, 'r') as f:
      mod_name = "nbox."+nbox_file.replace(code, '').replace('.py', '').replace('/', '.')
      _hash = U.hash_(f.read(), "sha256")
      if _hash != state["hashes"].get(mod_name, {}).get("code", ""):
        state["hashes"].setdefault(mod_name, {})["code"] = _hash
        mod_to_do.add(mod_name)
      mod_to_file[mod_name] = nbox_file

      # check if the src file for this file exists or not
      mod_src = src+f"code/{mod_name}.md"
      if not os.path.exists(mod_src):
        if any([p.match(mod_name) for p in ig_pats]):
          continue
        with open(mod_src, 'w') as f:
          t = f"# {mod_name}\n\nWrite your docs here.\n"
          f.write(t)
          state["hashes"].setdefault(mod_name, {})["src"] = U.hash_(t, "sha256")
        src_files.append(mod_src)

  deprecated_src = []
  for src_file in src_files:
    with open(src_file, 'r') as f:
      mod_name = src_file.replace(src + "code", '').replace('.md', '').replace('/', '.')[1:]
      if mod_name not in mod_to_file:
        deprecated_src.append(src_file)
        continue
      _hash = U.hash_(f.read(), "sha256")
      if _hash != state["hashes"].get(mod_name, {}).get("src", ""):
        state["hashes"].setdefault(mod_name, {})["src"] = _hash
        mod_to_do.add(mod_name)

  print(f"Total files to update: {len(mod_to_do)} ")

  # finally store the cache back in the file
  U.to_json(state, STATE_FILE)


def generate(
  code: str,
  src: str,
  gen: str,
  ignore_pat: List[str] = [],
  include_pat: List[str] = []
):
  """Process your src & code, generate final copies"""
  if ignore_pat and include_pat:
    raise ValueError("You can't use both ignore_pat and include_pat")

  # This function does not do anything with state management, it just generates all the files that are
  # common in the gen/code folder.
  nbox_files = {
    "nbox."+x.replace(code, '').replace('.py', '').replace('/', '.'): x
    for x in U.get_files_in_folder(code, ".py", False)
  }
  src_files = {
    x.replace(src + "code", '').replace('.md', '').replace('/', '.')[1:]: x
    for x in U.get_files_in_folder(U.join(src, "code"), ".md", False)
  }
  _nbxm = set(nbox_files.keys())
  _srcm = set(src_files.keys())
  _comm = _nbxm.intersection(_srcm)
  _only_nbx = _nbxm.difference(_srcm)
  _only_src = _srcm.difference(_nbxm)
  print("-------")
  print(f"Files in nbox/ folder: {len(nbox_files)}")
  print(f"Files in src/ folder: {len(src_files)}")
  print(f"Files in common: {len(_comm)}")
  print(f"Files in nbox/ only: {len(_only_nbx)}")
  print(f"Files in src/ only: {len(_only_src)}")
  print(f"Total files skipped: {len(_only_src) + len(_only_nbx)}")
  print("-------")

  for mod in _comm:
    _code = nbox_files[mod]; _src = src_files[mod]
    print(mod, _code, _src)


if __name__ == "__main__":
  current_folder = U.folder(__file__)
  print("Moving to directory:", current_folder)
  os.chdir(current_folder)

  CODE = "../nbox/"
  GEN = "gen/"
  SRC = "src/"

  assert os.path.exists(CODE) and os.path.isdir(CODE), "nbox folder not found"
  if not os.path.exists(GEN):
    print("Creating gen folder")
    os.mkdir(GEN)
  assert os.path.exists(SRC) and os.path.isdir(SRC), "src folder not found"

  # create the CLI
  comms = {
    "create_blanks": partial(create_blanks, CODE, SRC, GEN),
    "generate": partial(generate, CODE, SRC, GEN),
  }
  for k,v in comms.items():
    v.__doc__ = v.func.__doc__
  Fire(comms)
