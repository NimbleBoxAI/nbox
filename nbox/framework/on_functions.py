# weird file name, yessir. Why? because
#
# from nbox.framework.on_functions import PureFunctionParser
# read as "from nbox's framework on Functions import the pure-function Parser"

import inspect
from logging import getLogger
logger = getLogger("on_func")


class PureFunctionParser:
  def __init__(self, object):
    """This Parser takes a Python object's reference and creates a nbox-job for it.
    This part of the code will be built as we solve the problem of running scripts
    simply with performance gurantees, securely in an automated fashion on the
    NBX Build.

    raises:

      AssertionError: if the object is not callable
      AssertionError: if the object is not a function
    """
    self.object = object
    assert callable(object), "object must be a callable object"
    self.code = inspect.getsource(object)
    assert self.code.startswith("def "), "input_object must be a function"
    self.fn_name = self.code.split("(")[0].split(" ")[1]
    self.fn_filepath = inspect.getsourcefile(object)

    # next we 

    self.file_code = f'''
# This is nbox-jobs auto generated code

{self.code}
if __name__ == "__main__":
  import joblib
  with open("./data.pkl", "rb") as f:
    data = joblib.load(f)
  {self.fn_name}(**data)
'''.strip()

    logger.info(f"self.fn_name: {self.fn_name}")



