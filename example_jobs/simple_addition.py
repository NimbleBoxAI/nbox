# this is an absolute overkill for just adding two numbers but we will still do it

from nbox import Instance

instance = Instance("GPT4NBX", url = "https://test-2.nimblebox.ai")
# instance = Instance.create("GPT4NBX", url = "https://test-2.nimblebox.ai")

instance.start(True)
my_uid = instance("add.py")
print("My UID:", my_uid)
instance(my_uid)
