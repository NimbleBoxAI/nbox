from nbox import Operator

class ShellCommand(Operator):
  def __init__(self, *commands):
    """Run multiple shell commands, uses ``shelex`` to prevent injection"""
    super().__init__()
    import string

    self.commands = commands
    all_in = []
    for c in self.commands:
      all_in.extend([tup[1] for tup in string.Formatter().parse(c) if tup[1] is not None])
    self._inputs = all_in

  def forward(self, *args, **kwargs):
    import shlex
    import subprocess

    for comm in self.commands:
      comm = comm.format(*args, **kwargs)
      comm = shlex.split(comm)
      subprocess.run(comm, check = True)


class Python(Operator):
  def __init__(self, func, *args, **kwargs):
    """Convert a python function into an operator, everything has to be passed at runtime"""
    super().__init__()
    self.fak = (func, args, kwargs)

  def forward(self):
    return self.fak[0](*self.fak[1], **self.fak[2])


class PythonScript(Operator):
  def __init__(self, fpath, **kwargs):
    """Run any file in python as an operator"""
    super().__init__()
    self.fp = fpath
    self.kwargs = kwargs

  def forward(self):
    raise NotImplementedError("PythonScript is not implemented yet")

    import importlib.util
    spec = importlib.util.spec_from_file_location("script", self.fp)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script.main(**self.kwargs)


class PythonNotebook(Operator):
  def __init__(self, fpath, **kwargs):
    """Run any file in python as an operator"""
    super().__init__()
    self.fp = fpath
    self.kwargs = kwargs

  def forward(self):
    raise NotImplementedError("PythonNotebook is not implemented yet")

    import importlib.util
    spec = importlib.util.spec_from_file_location("notebook", self.fp)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    return notebook.main(**self.kwargs)


class GitClone(Operator):
  def __init__(self, url, path = None, branch = None):
    """Clone a git repository into a path"""
    super().__init__()
    self.url = url
    self.path = path
    self.branch = branch

  def forward(self):
    # git clone -b <branch> <repo> <path>
    import subprocess
    command = ["git", "clone"]
    if self.branch:
      command.append("-b")
      command.append(self.branch)
    if self.path:
      command.append(self.path)
    command.append(self.url)
    subprocess.run(command, check = True)

