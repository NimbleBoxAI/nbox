from nbox import Operator

from nbox.lib.arch import StepOp

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
      called_process: subprocess.CompletedProcess = subprocess.run(comm, check = True)
      if called_process.returncode != 0:
        raise Exception(f"Command {comm} failed with return code {called_process.stdout}")


class Python(Operator):
  def __init__(self, func, *args, **kwargs):
    """Convert a python function into an operator, everything has to be passed at runtime"""
    super().__init__()
    self.fak = (func, args, kwargs)

  def forward(self):
    return self.fak[0](*self.fak[1], **self.fak[2])


class PythonScript(StepOp):
  def __init__(self, fpath, **kwargs):
    """Run any file in python as an operator"""
    super().__init__()
    self.fp = fpath
    self.kwargs = kwargs

    kwargs_strings = []
    for k,v in kwargs.items():
      if v != None:
        kwargs_strings.append(f"--{k}={v}")
      else:
        kwargs_strings.append(f"--{k}")
    kwargs_string = "\ \n".join(kwargs_strings)
    self.add_step(ShellCommand(
      f"./venv/bin/python {fpath} {kwargs_string}"
    ))


class PythonNotebook(StepOp):
  def __init__(self, fpath):
    """Run any file in python as an operator"""
    super().__init__()
    self.fp = fpath
    self.add_step(ShellCommand(
      f"jupyter nbconvert --to notebook --execute {fpath}"
    ))


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

