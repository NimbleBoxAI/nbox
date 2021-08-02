import unittest
import aibox
from torchvision import models

from aibox import utils

class ImportTest(unittest.TestCase):

    def test_utils(self):

        cat_url = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        out = utils.get_image(cat_url)
        print(out)

    def test_mobilenetv2_url(self):
        model = aibox.load("mobilenetv2")
        image = utils.get_image("https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg")
        out = model(image)

    def test_mobilenetv2_filepath(self):
        import os
        model = aibox.load("mobilenetv2")
        image = r"./assets/cat.jpg"
        out = model(image)

if __name__ == '__main__':
    unittest.main()
