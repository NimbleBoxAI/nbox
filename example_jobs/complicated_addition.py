# this is an absolute overkill for just adding two numbers but we will still do it in a functional way
# using our API wrapped as python

from nbox import Instance
from pprint import pprint as peepee

instance = Instance("GPT4NBX", url = "https://test-2.nimblebox.ai")
# instance = Instance.create("GPT4NBX", url = "https://test-2.nimblebox.ai")

instance.compute_server.test("get")
instance.compute_server.get_files(
  "post", {"dir_path": "/"}, verbose = True
)
out = instance.compute_server.run_script("post", {"script": "add.py"}, verbose = True)
print("RUN:")
peepee(out)
uid = out["uid"]

out = instance.compute_server.get_script_status("post", {"uid": out["uid"]}, verbose = True)
print("STATUS:")
peepee(out)

out = instance.compute_server.clear_jobs_db("get", verbose = True)
print("CLEAR:")
peepee(out)

out = instance.compute_server.get_script_status("post", {"uid": uid}, verbose = True)
print("STATUS:")
peepee(out)
