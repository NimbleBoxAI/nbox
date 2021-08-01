import unittest
import aibox
from torchvision import models

class ImportTest(unittest.TestCase):

  def efficient_import(self):
  model = aibox.load("mobilenetv2")
  out = model("./nimblebox.png")
  print(out)

if __name__ == '__main__':
  unittest.main()
  