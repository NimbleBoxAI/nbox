# run your "pure function" remotely using nbox-jobs. This technology enables powerful
# new ways to use cloud compute on that can radically change the way we program systems.
#
# Simply put a "pure function" is a type of function that has:
# 1. all the required packages, imports and required functions write inside a the function
# code itself
# 2. the inputs and outputs must by pickle-able
#

from nbox import Instance

# read "from nbox's framework on Functions import the pure-function Parser"
from nbox.framework.on_functions import PureFunctionParser

def add_two_nos(a, b):
  return a + b

pfp = PureFunctionParser(add_two_nos)
# print(pfp.code)
print(pfp.file_code)


# instance = Instance.create("pure_function")
# instance(add_two_nos)
