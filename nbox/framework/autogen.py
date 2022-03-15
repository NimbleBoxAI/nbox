# this script is meant to be run to generate latest code for on_ml.py utility functions

import re
import jinja2
import inspect
from textwrap import indent
from datetime import datetime
from typing import Callable, List

from .. import utils as U
from ..utils import logger

class MLFrameworkRegistry:
  def __init__(self):
    self.stub_classes = {}
    self.conditional_fns = {}
    self.repr_string = []

  def __repr__(self) -> str:
    return f"< nbox.framework.{self.__class__.__qualname__}\n" + \
      indent("\n".join(self.repr_string), "  ") + \
      "\n>"

  def conditional(
    self,
    framework: str
  ):
    def _wrapper(fn):
      self.conditional_fns[framework] = fn
      return fn
    return _wrapper

  def register(
    self,
    stub_name,
    framework: str,
    target: str,
    message_name: str,
    export_fn_import: str,
    target_processor_name: str = None,
    dependencies: List[str] = [],
    ignore_args: List[str] = [],
    verbose = True,
  ):
    def _wrapper(processor_fn: Callable):
      if verbose:
        logger.debug(f"Registering {message_name} and {processor_fn.__name__}")
      
      # import the function if raises import error return None
      export_fn = re.findall(r"import (.*)", export_fn_import)[0]
      if verbose:
        logger.debug(f"export_fn: {export_fn}")
        logger.debug(f"export_fn_import: {export_fn_import}")
        self.repr_string.append(f"{message_name} -> {processor_fn.__name__}")
      try:
        exec(export_fn_import, locals())
      except ImportError:
        logger.error(f"ImportError: {export_fn_import}")
        return None
      export_fn = eval(f"{export_fn}", locals())

      # create the arguments for the template
      out = inspect.getfullargspec(export_fn)
      args = {}
      if out.defaults != None:
        reverse_dict = {}
        for k,v in zip(out.args[::-1], out.defaults[::-1]):
          reverse_dict[k] = v
        back = {k:v for k,v in reversed(reverse_dict.items())}
        for i in range(len(out.args) - len(out.defaults)):
          args[out.args[i]] = None
        args.update(back)
      else:
        for i in range(len(out.args)):
          args[out.args[i]] = None

      arg_strings = []
      final_args = []
      _ignored = {}
      for k,v in args.items():
        if k in ignore_args:
          _ignored[k] = v
          continue
        if re.findall(r"<(.*)>", repr(v)):
          # probably any python object like: <torch.jit.CompilationUnit object at 0x10b682c30>
          _ignored[k] = v
          continue
        if v != None and not isinstance(v, (bool, int, float, str)):
          v = None
        arg_strings.append(f"{k}={v}")
        final_args.append(k)

      template_data = dict(
        dependencies = dependencies,
        framework = framework,
        target = target,
        target_processor_name = target_processor_name,
        
        # message related
        message_name = message_name,
        arg_strings = arg_strings,
        doc = inspect.getdoc(export_fn),
        args = final_args,

        # stub related
        stub_name = stub_name,
        processor_fn = processor_fn.__name__,
      )
      self.stub_classes.setdefault(framework, []).append(template_data)
      return processor_fn
    return _wrapper

  def prepare(self,):
    prepared_frameworks = []
    for frm, template in self.stub_classes.items():
      cond_fn = self.conditional_fns.get(frm, "")
      if cond_fn:
        cond_fn = cond_fn.__name__
        
      data = {
        "messages": [],
        "stubs": [],
        "dependencies": [],
      }
      for template_data in template:
        data["messages"].append({
          "message_name": template_data["message_name"],
          "arg_strings": template_data["arg_strings"],
          "doc": template_data["doc"],
          "args": template_data["args"],
          "dependencies": template_data["dependencies"],
        })
        data["stubs"].append({
          "stub_name": template_data["stub_name"],
          "processor_fn": template_data["processor_fn"],
          "message_name": template_data["message_name"],
          "dependencies": template_data["dependencies"],
          "target": template_data["target"],
          "target_processor_name": template_data["target_processor_name"],
        })
      prepared_frameworks.append((data, frm, cond_fn))
    return prepared_frameworks

# create a register where we can store all the methods
ml_register = MLFrameworkRegistry()

# compiler function
def compile():
  src_file =  U.join(U.folder(U.folder(__file__)), "assets", "ml.jinja")
  trg_file = U.join(U.folder(__file__), "ml.py") # ml.py in this folder
  data = ml_register.prepare()

  # for (data, frm) in data:
  #   print(type(data))
  #   for (data, frm) in data:
  #     print(frm)

  with open(src_file, "r") as src, open(trg_file, "w") as trg:
    trg.write(
      jinja2.Template(src.read()).render(
        timestamp = datetime.utcnow().isoformat(),
        frameworks = data,
        zip = zip,
    ))
