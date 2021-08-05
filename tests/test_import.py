import unittest
from torchvision import models

import nbox
from nbox import utils

class ImportTest(unittest.TestCase):

    def test_utils(self):
        cat_url = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        out = utils.get_image(cat_url)
        print(out)

if __name__ == '__main__':
    unittest.main()
