import os
os.environ["NBOX_LOG_LEVEL"] = "info"
os.environ["NBOX_NO_AUTH"] = "1"
os.environ["NBOX_NO_LOAD_GRPC"] = "1"
os.environ["NBOX_NO_LOAD_WS"] = "1"
os.environ["NBOX_NO_CHECK_VERSION"] = "1"

import re
import json
import jinja2
from fire import Fire
from typing import List
from subprocess import Popen
from functools import partial
from pprint import pprint as pp
from docstring_parser import parse as parse_docstring
from docstring_parser.common import DocstringStyle

import nbox
from nbox.utils import folder, logger, get_files_in_folder, threaded_map
from nbox.nbxlib import astea
from nbox.nbxlib.astea import Astea, IndexTypes


def doc_str_to_mdx(
  docstring: str,
  fn_name: str = "",
  code_folder: str = "",
  code_prefix: str = "",
  docs_code_loc: str = "/code/autogen",
  source_fp: str = "",
  source_lineno: int = 0,
):
  """
  Args:
    docstring (str): docstring for function or class
    fn_name (str): name of the function, if provided adds a H2 element
    docs_code_loc (str, optional): the prefix of code text path, this is path inside "public" folder
  """
  # this function has following steps:
  # 01 parse and get markdown objects
  # 02 take code pieces and return code text strings
  # 03 add proper tags to the markdown elements like {% .margint %}
  # print(docstring)
  # doc_str = parse_docstring(docstring, style = DocstringStyle.GOOGLE)
  doc_str = parse_docstring(docstring)

  # markdown part
  md_part = ""
  if doc_str.short_description:
    md_part += doc_str.short_description + " " + "\n"*int(doc_str.blank_after_short_description)
  if doc_str.long_description:
    md_part += doc_str.long_description + " " + "\n"*int(doc_str.blank_after_long_description)
  md_lines = md_part.splitlines()
  for i, line in enumerate(md_lines):
    if line.startswith("#"):
      md_lines[i] = line + " {% .margint8 %}"
  md_part = "\n".join(md_lines)
  
  # process the code piece
  code = ""
  for x in doc_str.examples:
    code = x.description.replace(">>> ", "")
    code = code.replace("... ", "# ")

  code_fp = f"{docs_code_loc}/{code_prefix}{fn_name}.txt"
  if code and code_folder:
    local_loc = f"{code_folder}/{code_prefix}{fn_name}.txt"
    with open(local_loc, "w") as f:
      f.write(code)

  # print(doc_str.__dict__)
  source = f"https://github.com/NimbleBoxAI/nbox/blob/master/nbox/{source_fp}.py#L{source_lineno}"
  params = []
  for p in doc_str.params:
    # {"name":"method","type":"str","description":"This is description"}
    params.append({
      "name": p.arg_name,
      "type": str(p.type_name),
      "description": p.description,
    })

  # template this thing
  container = f'{{% VisionContainer label="{fn_name}" source="{source}" variant="function" params={json.dumps(params)}/%}}'
  fn_doc = '''{{ container }}
{!% if md_part %!}\n{{ md_part }}{!% endif %!}
{!% if codelines %!}
**Example**:
{% Code languages=["Python"] codeBlock=["{{ code_fp }}"] height={{ 14 * min(20, len(codelines)) }} /%}{!% endif %!}
{!% if returns %!}**Returns**: {!% for a in returns %!}
  - `{{ a.type_name }}`: {{ a.description }}{!% endfor %!}
{!% endif %!}
'''.strip()
  fn_doc = jinja2.Template(
    fn_doc,
    block_start_string = '{!%',
    block_end_string = '%!}',
  ).render(
    container = container,
    md_part = md_part,
    codelines = code.splitlines(),
    returns = doc_str.many_returns,
    code_fp = code_fp,
    len = len,
    min = min,
  )

  return fn_doc


def default_template(
  mod: str,
  nbox_folder: str,
  code_folder: str,
):
  """Create a default documentation page for a module."""
  fp = ".".join(mod.split(".")[1:]).replace('.','/')
  code = f"{nbox_folder}{fp}.py"
  tea = Astea(code)
  # print(mod, fp)

  # `{{ tea.name.strip('./').replace('/','.') }}` {% .marginb8 %}
  # {# The default template first goes over all the functions then goes over all the classes and it's functions #}
  # Functions
  _doc_str_to_mdx = partial(doc_str_to_mdx, code_folder=code_folder, source_fp=fp)

  # get all the generations for the classes
  classes_md = ""
  for x in tea.find(types = IndexTypes.CLASS):
    class_index = x.find(types = IndexTypes.FUNCTION)
    # print(class_index)
    # if mod == "nbox.operator":
      # print(x, class_index)
    # print(mod, fp, x.name, x.node.lineno)
    source = f"https://github.com/NimbleBoxAI/nbox/blob/master/nbox/{fp}.py#L{x.node.lineno}"
    container = f'{{% VisionContainer label="{x.name}" source="{source}" variant="class" /%}}'
    _t = jinja2.Template(
      '''{{ container }}
{!% if class_index %!}{!% for x in class_index %!}
{{ doc_str_to_mdx(x.docstring(), x.name, source_lineno = x.node.lineno) }}
{!% endfor %!}{!% endif %!}
      '''.strip(),
      block_start_string = '{!%',
      block_end_string = '%!}',
    )
    classes_md += _t.render(
      container = container,
      class_index = class_index,
      doc_str_to_mdx = partial(_doc_str_to_mdx, code_prefix = f"{x.name}."),
      cls = x,
    )

  # get all the generations for the functions
  _t = jinja2.Template(
    '''
{!% for fn in tea.find(types=IndexTypes.FUNCTION) %!}
{{ doc_str_to_mdx(fn.docstring(), fn.name, source_lineno = fn.node.lineno) }}
{!% endfor %!}
    ''',
    block_start_string = '{!%',
    block_end_string = '%!}',
  )
  functions_md = _t.render(
    tea = tea,
    doc_str_to_mdx = _doc_str_to_mdx,
    IndexTypes = IndexTypes,
  ).strip()
  
  # create the final template
  # print(tea.docstring())
  template = '''# {{ mod }} {% .marginb8 %}
{{ tea.docstring() }}

{!% if functions_md %!}# Functions {% .margint8 %}

{{ functions_md }}{!% endif %!}

{!% if classes_md %!}# Classes {% .margint8 %}

{{ classes_md }}{!% endif %!}
    '''
  template = jinja2.Template(
    template,
    block_start_string = '{!%',
    block_end_string = '%!}',
  ).render(
    tea = tea,
    mod = mod,
    functions_md = functions_md,
    classes_md = classes_md,
  ).strip()

  return template



def module_to_mdx(
  mod: str = "nbox.nbxlib.astea",
  target_folder: str = "gen/",
  code_folder: str = "autogen/",
  nbox_folder: str = "../nbox/",
  template: str = ""
):
  """
  Each template will get `astea`, `tea`, `IndexTypes` and `doc_str_to_mdx`.
  
  Args:
    mod (str): name of the module for which documentation is to be generated
    target_folder(str): the final mdx file will be stored inside this folder
    code_folder (str): the text files for corresponding code items will be stored in this folder
    nbox_folder (str): the folder where nbox is installed
    template (str, optional): If provided will use this template, else will use the default one
  """
  if not template:
    # use the default template
    out = default_template(mod = mod, nbox_folder = nbox_folder, code_folder = code_folder)
  else:
    with open(template, "r") as src:
      template = src.read()  

    # load the template
    T = jinja2.Template(
      template,
      block_start_string = '{!%',
      block_end_string = '%!}',
    )
    fp = ".".join(mod.split(".")[1:]).replace('.','/')
    code = f"{nbox_folder}{fp}.py"
    tea = Astea(code)
    out = T.render(
      tea = tea,
      astea = astea,
      IndexTypes = IndexTypes,
      doc_str_to_mdx = partial(doc_str_to_mdx, code_folder = code_folder),
    )

  trg_fp = f"{target_folder}{mod}".replace(".","/").replace("/nbox", "")
  os.makedirs(os.path.dirname(trg_fp), exist_ok=True)
  with open(trg_fp+".md", "w") as f:
    f.write(out)


def main(ignore: List[str] = [], v: bool = False):
  """Generate documentation for nbox
  
  Args:
    ignore (List[str], optional): ignore these patterns while generating documentation
  """
  current_folder = folder(__file__)
  logger.info(f"Moving to directory: {current_folder}")
  os.chdir(current_folder)

  NBOX = nbox.__path__[0]
  GEN = "nbox_gen/"
  CODE = "autogen/"
  SRC = "src/"

  if not os.path.exists(GEN):
    logger.info(f"Creating gen folder: {GEN}")
    os.mkdir(GEN)
  else:
    logger.info(f"Clearing gen folder: {GEN}")
    Popen(f"rm -rf {GEN}*", shell=True).wait()
  os.mkdir(f"{GEN}/docs")

  if not os.path.exists(CODE):
    logger.info(f"Creating autogen folder: {CODE}")
    os.mkdir(CODE)
  else:
    logger.info(f"Clearing autogen folder: {CODE}")
    Popen(f"rm -rf {CODE}*", shell=True).wait()
  
  if not os.path.exists(SRC):
    logger.warning(f"Could not find {SRC} folder, all the templates will be autogenerated")
    src_files = set()
  else:
    src_files = set(sorted(get_files_in_folder(SRC, ".md", False)))
    logger.info(f"Found {len(src_files)} files in {SRC}")

  nbox_files = sorted(get_files_in_folder(NBOX, ".py", False))
  logger.info(f"Found the {len(nbox_files)} files in nbox folder")

  # create regex patterns for ignored files
  ig_pats = []
  for p in ignore:
    pat = ""
    if not p.startswith("^"):
      pat += "^.*"
    pat += p
    if not p.endswith("$"):
      pat += ".*$"
    # print(pat)
    ig_pats.append(re.compile(pat))

  # create the data for each module to be generated
  module_data = []
  for f in nbox_files:
    mod_name = "nbox"+f.replace(NBOX, '').replace('.py', '').replace('/', '.')
    if any([p.match(mod_name) for p in ig_pats]):
      continue
    template = f"{SRC}docs/{mod_name}.md"
    if not os.path.exists(template):
      template = ""
    else:
      src_files.remove(template)
    data = {
      "mod": mod_name,
      "target_folder": f"{GEN}docs/", 
      "code_folder": CODE,
      "nbox_folder": NBOX + "/",
      "template": template
    }
    _values = tuple(data.values())
    # print(_values)
    module_data.append(_values)

  # some code to test for a specific module
  # # mod = "nbox.nbxlib.astea"
  # mod = "nbox.subway"
  # module_data = [m for m in module_data if m[0] == mod]
  # module_to_mdx(*module_data[0])
  
  _ = threaded_map(module_to_mdx, module_data)

  # all the files in src_files are those that simply need to be copied to the gen folder
  for f in src_files:
    child_folders = f.split('/')[1:]
    if len(child_folders) > 1:
      child_folders = child_folders[:-1]
      child_folders = "/".join(child_folders)
      if not os.path.exists(f"{GEN}{child_folders}"):
        os.makedirs(f"{GEN}{child_folders}")
    trg = f"{GEN}{'/'.join(f.split('/')[1:])}"
    if v:
      print("Copying", f, "to", trg)
    Popen(["cp", f, trg]).wait()

  # next we need to create nboxRoutes.json which is required by our docs, the GEN folder has everything
  # in a flat structure, we need to create a tree structure. We can do this by putting all the files that
  # start with "nbox" in "API Docs" and others based on the '.' in the name.

  routes = {"routes": []}
  gen_files = sorted(get_files_in_folder(GEN+"docs/", ".md", False))
  if v:
    print(gen_files)
  for f in gen_files:
    f = f[len(GEN):].replace(".md", "")
    parts = f.split("/")[1:]
    if v:
      print(parts)
    if len(parts) == 1:
      f = "/".join(parts)
      routes["routes"].append({
        "label": f,
        "path": '/nbox/docs/' + f,
      })
    else:
      # we need to create the tree structure
      parent = routes
      for p in parts[:-1]:
        found = False
        for r in parent["routes"]:
          if r["label"] == p:
            parent = r
            found = True
            break
        if not found:
          parent["routes"].append({
            "label": p,
            "routes": []
          })
          parent = parent["routes"][-1]
      parent["routes"].append({
        "label": parts[-1],
        "path": '/nbox/' + f,
      })

  if v:
    pp(routes["routes"])

  # update the predefined routes object
  with open(f"nboxRoutes.json", "r") as f:
    data = json.load(f)
  for r in data["routes"]:
    if r["label"] == "API Reference":
      # we are only going to update the inner layer
      r["routes"] = routes["routes"]#[0]["routes"]
      break
  with open(f"nboxRoutes.json", "w") as f:
    f.write(json.dumps(data, indent=2))


if __name__ == "__main__":
  Fire(main)
