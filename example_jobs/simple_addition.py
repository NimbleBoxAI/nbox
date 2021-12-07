# # this is an absolute overkill for just adding two numbers but we will
# # still do it

import time

from pprint import pprint as peepee

from nbox import Instance
from nbox.utils import nbox_session

instance = Instance("GPT4NBX", url = "https://test-2.nimblebox.ai")
# instance = Instance.create("GPT4NBX", url = "https://test-2.nimblebox.ai")
instance.start(True)
# instance("add.py")
# instance.stop()

instance.compute_server.test("get").json()
instance.compute_server.get_files(
  "post", {"dir_path": "/"}, verbose = True
).json()
out = instance.compute_server.run_script("post", {"script": "add.py"}, verbose = True).json()
print("RUN:")
peepee(out)
uid = out["uid"]

out = instance.compute_server.get_script_status("post", {"uid": out["uid"]}, verbose = True).json()
print("STATUS:")
peepee(out)

out = instance.compute_server.clear_jobs_db("get", verbose = True).json()
print("CLEAR:")
peepee(out)

out = instance.compute_server.get_script_status("post", {"uid": uid}, verbose = True).json()
print("STATUS:")
peepee(out)

# class Sample():
#   def __init__(self):
#     pass
#   def __getattribute__(self, __name: str):
#     print(__name)
#     def __func(a, b):
#       return a + b
#     return __func

# s = Sample()
# print(s.add)
# print(s.add(123, 123))
# print(s.multiply(123, 123))
