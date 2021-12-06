# this is an absolute overkill for just adding two numbers but we will
# still do it

from nbox import Instance
from nbox.utils import nbox_session

instance = Instance("GPT4NBX", url = "https://test-2.nimblebox.ai")
print(instance)
# instance.test()
# instance.start(True, wait = True)
# instance.get_files()
instance.run_script("add.py")
instance.get_script_status("add.py")

# instance.stop(wait = True)
