"""
Jupyter Notebooks are user by millions of developers world wide, so we said why not just
create an operator that runs the jupyter notebook and creates a python file out of it.
Catch: Currently we do not support notebooks that have a `!` for shell commands, please
replace them with `subprocess.call(shelex.split(<com>))`

Example:

```
from nbox.lib.notebook import NotebookRunner
op = NotebookRunner("./sample.ipynb")
op() # call and it will do the rest
```
"""

import sys
import ast
import json
import importlib.util
from functools import lru_cache

import nbox.utils as U
from nbox import operator, Operator, logger
from nbox.lib.shell import ShellCommand

@lru_cache(maxsize=1)
def get_py_builtin_all():
  # a cheap 99% solution that works for 99% of people
  builtin_path = U.join(U.folder(U.folder(__file__)), "assets/builtin")
  with open(builtin_path, "r") as f:
    items = []
    for l in f:
      if not l.startswith("#"):
        items.append(l.strip())
  return items

def get_package_version(name = 'tensorflow'):
  # https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed

  # I don't know how to remove some of the packages which are not part of builtins
  # but are still in the python lib like json or logging.

  if name in sys.builtin_module_names or name in get_py_builtin_all():
    return ""

  if name in sys.modules:
    version = getattr(sys.modules[name], "__version__", None)
    if version:
      return version
    return ""
  spec = importlib.util.find_spec(name)
  if spec is not None:
    # If you choose to perform the actual import ...
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    version = getattr(sys.modules[name], "__version__", None)
    del(sys.modules[name])
    if version:
      return version
    return ""

  # TODO: when can't find a package -> maybe it is a file relative to the notebook
  return False

class ImportNodeVisitor(ast.NodeVisitor):
  def __init__(self):
    self.import_modules = set()

  def visit_Import(self, node: ast.Import):
    for n in node.names:
      n = n.name.split(".")[0]
      self.import_modules.add(n)

  def visit_ImportFrom(self, node: ast.ImportFrom):
    self.import_modules.add(node.module)


def process_codeblocks(cells) -> str:
  """This function is responsible for unpacking the code blocks and creating the python file from it."""
  code_lines = []
  for cell in cells:
    if not cell["cell_type"] == "code":
      # this is not even code, probably markdown, so skip
      # ideally I would like to write down everything, but
      # it's okay here because that doesn't matter to computer
      # maybe an AI can read it and do something.
      continue
    code_lines.extend(cell["source"])
    code_lines.append("\n")

  # let's make this shit tricky:
  # Jupyter has support for built in shell commands by adding a ! like this
  #     !echo "hello"
  # moreover it has support for this in loops and all with reference to actual
  # python values by adding a $ in front of the variable name
  #     for x in range(2):
  #       !echo "alpha-$x"
  # since things like this cannot be really parsed by python ast, we need to
  # perform a modifications to the code before we parse it
  # remember this can also be in multiple lines by adding \ at the end
  #
  # this is a very hard problem to solve because of the number of edge cases
  # if there is a ! inside triple quotes, then it cannot be updated
  # however it is very hard to determine the triple quoted strings without
  # running the code. the chicken and egg problem
  # so for now we are saying if you have that kindly remove it.

  # jupyter notebook also has support for magic commands through % and %% like
  # %matplotlib inline
  # %%time
  # so we need to remove those as well, the good thing is that they cannot persist
  # across multiple lines, so we can just remove them line by line
  for i, line in enumerate(code_lines):
    if line.startswith("%"):
      line = "# " + line
      code_lines[i] = line
  full_code_raw = "".join(code_lines)

  # go over all the imports and sort them out
  nv = ImportNodeVisitor()
  try:
    nv.visit(ast.parse(full_code_raw))
  except Exception as e:
    # str(e) => (<unknown>, line 187) # get the 187
    eline = int(str(e)[:-1].split("line")[1].strip())
    line = code_lines[eline-1]
    if line.strip().startswith("!"):
      # if the line is commented out, then it's okay
      logger.error("Error:")
      logger.error(" reason: this is supported by jupyter notebook but not python")
      logger.error("    fix: subprocess.call(shelex.split(<com>))")
      logger.error(f"    >>> {line}")
      raise
    else:
      raise e
  import_modules = nv.import_modules
  module_to_version = {}
  for x in import_modules:
    v = get_package_version(x)
    if v:
      module_to_version[x] = v

  # there is no need to brong the imports to top because scoping will be unpredictable
  return full_code_raw, module_to_version

@operator()
def nb2script(nb, path: str = None):
  """This function takes in a notebook file and creates a python file from it.
  
  Args:
    nb (str): path to the notebook file
    path (str): path to the python file to be created, if `None` same name as notebook
  """
  # read the file and process it to get the code
  with open(nb, "r") as f:
    code = json.load(f)
  full_code_raw, import_modules = process_codeblocks(code["cells"])
  if not full_code_raw:
    raise ValueError("Oops! No code blocks found in the notebook")

  # Start populating the header metadata and the complete the file
  kernel = "N/A"
  nb_version = f"{code['nbformat']}.{code['nbformat_minor']}"
  meta = code["metadata"]
  if meta:
    if "kernelspec" in meta:
      kernel = meta["kernelspec"]["display_name"]

  import_string = "\n".join(["# " + x for x in json.dumps(import_modules, indent=2).splitlines()])
  _metadata = f'''
# -------------
# Here's some information on the notebook:
#   NB Version: {nb_version}
#         Path: {nb}
#       Kernel: {kernel}
#
# Here's a list of found imports:
{import_string}
# -------------
  '''.strip()

  py_file = f'''
# Auto-generated by Nb2Script Operator: Do Not Modify
{_metadata}

# START >
{full_code_raw}
# < END
'''

  # write the file to correct location
  if path is None:
    path = nb.replace(".ipynb", ".py")
  with open(path, "w") as f:
    f.write(py_file)

  return import_modules


def update_requirements(req_path, mod_versions):
  lines = []
  mods_added = set()
  with open(req_path, "r") as f:
    for line in f:
      if line.startswith("#"):
        lines.append(line)
      else:
        mod = line.split("==")[0]
        if mod in mod_versions:
          ver = mod_versions[mod]
          line = f"\n{mod}"
          if ver:
            line += f"=={ver}"
          lines.append(line + "\n")
          mods_added.add(mod)
        else:
          lines.append(line)
  for mod, ver in mod_versions.items():
    if mod not in mods_added:
      ver = mod_versions[mod]
      line = f"\n{mod}"
      if ver:
        line += f"=={ver}"
      lines.append(line + "\n")
  with open(req_path, "w") as f:
    f.write("".join(lines))


class NotebookRunner(Operator):
  def __init__(self, nb: str, outfile: str = "out.py"):
    """Run any notebook as a python file.
    
    Args:
      nb (str): path to the notebook file
      outfile (str): path to the python file to be created
    """
    super().__init__()
    self.nb = nb

    # create the target runner file
    imports = nb2script(self.nb, outfile)

    # now update the imports of this file
    update_requirements("./requirements.txt", imports)

    # define the shell command to run this thing
    self.run = ShellCommand(f"/job/venv/bin/python3 {outfile}")

  def forward(self):
    self.run()
