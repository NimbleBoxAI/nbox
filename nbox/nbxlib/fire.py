import sys

class TC:
  # terminal colors from https://stackoverflow.com/a/287944/1123955
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
  us well for many years before we ended up here.
  
  - can load any kind of python object
  - '-' can be as many as you want they would be ignored
  - booleans are parsed as `--flag`
  - strings/int are parsed as `value`
  - /dict/list are parsed as `'["0", 123]'` or `"{a: 1, b: 2}"`
  """
  def help_for_comp(self, comp):
    lines = [
      f'{TC.FAIL}ERROR:{TC.ENDC} Cannot process empty command',
      f'{TC.OKGREEN}USAGE:{TC.ENDC} nbx COMMAND\n',
      'Command can be any of the following:\n',
      '\n'.join([f'  {c}' for c in dir(comp) if not c.startswith('_')]),
    ]
    self._print_and_exit(lines)

  def __init__(self, component):
    # component is how we are representing the object we are working with
    
    # get args and if nothing is passed then run the one time help which is outside the main loop
    args = sys.argv[1:]
    if len(args) == 0:
      lines = [
        f'{TC.FAIL}ERROR:{TC.ENDC} Cannot process empty command',
        f'{TC.OKGREEN}USAGE:{TC.ENDC} nbx COMMAND\n',
        'Command can be any of the following:\n',
        '\n'.join([f'  {c}' for c in dir(component) if not c.startswith('_')]),
      ]
      self._print_and_exit(lines)

    print(args)

    # # this is the args kwargs that will be refreshed every time we go inside a component
    # kwargs = {}
    # args_list = []
    # curr_ptr = 0
    # next_ptr = 0
    # while curr_ptr < len(args):
    #   arg = args[curr_ptr]
    #   if arg.startswith('-'):
    #     arg = arg.lstrip('-')      # remove all the dashes
    #     if arg in dir(component):  # if the arg is a method of the component
          

    
    
    # all_args = []

    # # what will be passed to the function, defined up here in case of partials
    # for i, arg in enumerate(args[1:]):
    #   # we need to do a bunch of analysis of the comp here and then based on those we need to create the required
    #   # data for the different templates
    #   # ex: if the comp is a partial function then we need to get functions details from the underlying function
    #   if type(comp) == partial:
    #     args_list.extend(tuple(comp.args))
    #     kwargs.update(comp.keywords)
    #     comp = comp.func

    #   if arg == '--help':
    #     usage = comp.__doc__
    #     # and this is a class then load the docstring from the __init__
    #     if usage is None and type(comp) == type:
    #       usage = comp.__init__.__doc__

    #     # make lines and exit
    #     lines = [
    #       f"{TC.BOLD}{TC.OKGREEN}Usage:{TC.ENDC} {os.path.basename(sys.argv[0])} {service_level}\n",
    #       f"{TC.BOLD}{TC.OKGREEN}Description:{TC.ENDC} {usage}\n",
    #       f"{TC.BOLD}{TC.OKGREEN}Options:{TC.ENDC}"
    #     ] + [f"  {x}" for x in dir(comp) if not x.startswith('_')]
    #     self._print_and_exit(lines)
      
    #   # get the comp and build the kwargs dict
    #   _comp = getattr(comp, arg, None)
    #   if _comp is None:
    #     # this means the user is probably trying to call this python object so we just break here
    #     all_args = args[i+1:]
    #     break

    #   # overwrite the comp
    #   comp = _comp

    # # by now we have the comp all the things required to run the comp, however they are not in the right order
    # # or all the keys provided are useful or not. here's a list of valid keys
    # # nbx jobs upload 'xyz' -id 'abc def' --bool -u '....'
    # # this is a tricky piece of code to write, so we will use a simple while loop with O(1) just like how
    # # language parsers work.
    # next_arg = ""
    # arg = all_args.pop(0)
    # while arg:
    #   if len(all_args):
    #     next_arg = all_args.pop(0)
    #   else:
    #     next_arg = None
    #   print(arg, next_arg, kwargs)

    #   if arg.startswith('-'):
    #     # this is a flag, this can be of two types one with = in it and one without
    #     if '=' in arg:
    #       # this is a flag with = in it
    #       key, value = arg.split('=')
    #       kwargs[key.strip('-')] = value
    #     else:
    #       arg = arg.strip('-')
    #       if next_arg.startswith('-'):
    #         kwargs[arg] = True # boolean
    #       else:
    #         kwargs[arg] = next_arg
    #       if len(all_args):
    #         arg = all_args.pop(0)
    #       else:
    #         break
    #       continue
    #   else:
    #     if kwargs:
    #       # this means we have a positional argument and a keyword argument which can either be a mistake
    #       # or an object being initialised and then its attributes being set used
    #       try:
    #         comp = self._init_comp(comp, *args_list, **kwargs)
    #         comp = getattr(comp, arg, None)
    #         if comp is None:
    #           lines = [
    #             f"{TC.BOLD}{TC.FAIL}ERROR:{TC.ENDC} Could not find command: '{arg}'. Available commands are:\n"
    #           ] + [f"  {x}" for x in tuple(comp.__dir__()) if not x.startswith('_')]
    #           self._print_and_exit(lines)
            
    #         # a fresh args kwargs
    #         args_list = []
    #         kwargs = {}
    #       except:
    #         lines = [
    #           f"{TC.BOLD}{TC.FAIL}ERROR:{TC.ENDC} Could not find command: '{arg}'. Available commands are:\n"
    #         ] + [f"  {x}" for x in tuple(comp.__dir__()) if not x.startswith('_')]
    #         self._print_and_exit(lines)
    #     else:
    #       args_list.append(arg)
    #     # arg = all_args.pop(0)

    #   arg = next_arg
    
    # print(kwargs, args_list, arg, all_args)

    # comp(*args_list, **kwargs)

  def _print_and_exit(self, lines):
    text = '\n'.join(lines) + '\n'
    # More(text, out=sys.stderr)
    print(text, file=sys.stderr)
    sys.exit(1)

  def _init_comp(self, comp, *args, **kwargs):
    return comp(*args, **kwargs)

  def More(self, contents: str, out):
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

