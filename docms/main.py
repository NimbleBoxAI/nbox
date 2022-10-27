import os
os.environ["NBOX_LOG_LEVEL"] = "info"
os.environ["NBOX_NO_AUTH"] = "1"
os.environ["NBOX_NO_LOAD_GRPC"] = "1"
os.environ["NBOX_NO_LOAD_WS"] = "1"
os.environ["NBOX_NO_CHECK_VERSION"] = "1"

import re
import jinja2
from fire import Fire
from typing import List
from subprocess import Popen
from functools import partial
from docstring_parser import parse as parse_docstring

import nbox
from nbox.utils import folder, logger, get_files_in_folder
from nbox.nbxlib import astea
from nbox.nbxlib.astea import Astea, IndexTypes

def doc_str_to_mdx(
  docstring: str,
  fn_name: str = "",
  code_folder: str = "",
  code_prefix: str = "/code/autogen",
):
  """
  Args:
    docstring (str): docstring for function or class
    fn_name (str): name of the function, if provided adds a H2 element
    code_prefix (str, optional): the prefix of code text path, this is path inside "public" folder
  """
  # this function has following steps:
  # 01 parse and get markdown objects
  # 02 take code pieces and return code text strings
  # 03 add proper tags to the markdown elements like {% .margint %}
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

  code_fp = f"{code_prefix}/{fn_name}.txt"
  if code and code_folder:
    with open(f"{code_folder}/{fn_name}.txt", "w") as f:
      f.write(code)
  
  # template this thing
  fn_doc = '''{!% if fn_name %!}## function `{{ fn_name }}` {% .margint8 %}{!% endif %!}
{!% if md_part %!}\n{{ md_part }}{!% endif %!}
{!% if codelines %!}
**Example**:
{% Code languages=["Python"] codeBlock=[{{ code_fp }}] height={{ 14 * min(20, len(codelines)) }} /%}{!% endif %!}
**Arguments**: {!% if not args %!}this function does not take in any arguments.{!% else %!} {!% for a in args %!}
  - {{ a.arg_name }} (`{{ a.type_name }}`{!% if not a.is_optional %!}, **required**{!% endif %!}): {{ a.description }}{!% endfor %!}
{!% endif %!}
**Returns**: {!% if not returns %!}this function does not return anything.{!% else %!} {!% for a in returns %!}
  - `{{ a.type_name }}`: {{ a.description }}{!% endfor %!}
{!% endif %!}
'''.strip()
  fn_doc = jinja2.Template(
    fn_doc,
    block_start_string = '{!%',
    block_end_string = '%!}',
  ).render(
    fn_name = fn_name,
    md_part = md_part,
    codelines = code.splitlines(),
    args = doc_str.params,
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

  # `{{ tea.name.strip('./').replace('/','.') }}` {% .marginb8 %}
  # {# The default template first goes over all the functions then goes over all the classes and it's functions #}
  # Functions

  _doc_str_to_mdx = partial(doc_str_to_mdx, code_folder=code_folder)
  # get all the generations for the classes
  classes_md = ""
  for x in tea.find(types = IndexTypes.CLASS):
    class_index = x.find(types = IndexTypes.FUNCTION)
    # if mod == "nbox.operator":
    #   print(x, class_index)
    _t = jinja2.Template(
      '''{!% if class_index %!}## class `{{ cls.name }}` {% .margint8 %}
{!% for x in class_index %!}
{{ doc_str_to_mdx(x.docstring(), x.name) }}
{!% endfor %!}{!% endif %!}
      '''.strip(),
      block_start_string = '{!%',
      block_end_string = '%!}',
    )
    classes_md += _t.render(
      class_index = class_index,
      doc_str_to_mdx = _doc_str_to_mdx,
      cls = x,
    )

  # if mod == "nbox.operator":
  #   print(classes_md)

  # get all the generations for the functions
  _t = jinja2.Template(
    '''
{!% for fn in tea.find(types=IndexTypes.FUNCTION) %!}
{{ doc_str_to_mdx(fn.docstring(), fn.name, ) }}
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
  template = '''# {{ tea.name.strip('./').replace('/','.') }} {% .marginb8 %}

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

  with open(f"{target_folder}{mod}.md", "w") as f:
    f.write(out)


def main(ignore: List[str] = []):
  """Generate documentation for nbox
  
  Args:
    ignore (List[str], optional): ignore these patterns while generating documentation
  """
  current_folder = folder(__file__)
  logger.info(f"Moving to directory: {current_folder}")
  os.chdir(current_folder)

  NBOX = nbox.__path__[0]
  GEN = "nbox_gen/"
  CODE = "nbox_autogen/"
  SRC = "src/"

  if not os.path.exists(GEN):
    logger.info(f"Creating gen folder: {GEN}")
    os.mkdir(GEN)
  if not os.path.exists(CODE):
    logger.info(f"Creating autogen folder: {CODE}")
    os.mkdir(CODE)
  
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
      "target_folder": GEN, 
      "code_folder": CODE,
      "template": template
    }
    module_data.append(data)
    print(data)

    # generate the file
    module_to_mdx(**data)

  # all the files in src_files are those that simply need to be copied to the gen folder
  for f in src_files:
    Popen(["cp", f, f"{GEN}{f.split('/')[-1]}"]).wait()


if __name__ == "__main__":
  Fire(main)
