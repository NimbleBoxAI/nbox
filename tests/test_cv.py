import unittest
from torchvision import models

import nbox
from nbox import utils

class ImportComputerVision(unittest.TestCase):

    def test_mobilenetv2_url(self):
        model = nbox.load("torchvision/mobilenetv2")
        image = utils.get_image("https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg")
        out = model(image)

    def test_mobilenetv2_filepath(self):
        import os
        model = nbox.load("torchvision/mobilenetv2")
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model(image)

    def test_mobilenetv2_numpy(self):
        import numpy as np
        model = nbox.load("torchvision/mobilenetv2")
        image = np.random.randint(low = 0, high = 256, size = (224, 224, 3)).astype(np.uint8)
        out = model(image)

    def test_mobilenetv2_url_batch(self):
        model = nbox.load("torchvision/mobilenetv2")
        image = utils.get_image("https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg")
        out = model([image, image])

    def test_mobilenetv2_filepath_batch(self):
        import os
        model = nbox.load("torchvision/mobilenetv2")
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model([image, image])

    def test_mobilenetv2_numpy_batch(self):
        import numpy as np
        model = nbox.load("torchvision/mobilenetv2")
        image = np.random.randint(low = 0, high = 256, size = (224, 224, 3)).astype(np.uint8)
        out = model([image, image])

if __name__ == '__main__':
    unittest.main()
