import os
import unittest

import nbox
from nbox import utils

# we check forward pass works and that results are reproducible
URL_CAT_TARGET_LIST = [862, 644, 626, 470, 827]
ASSET_CAT_TARGET_LIST = [862, 626, 470, 644, 556]

class ImportComputerVision(unittest.TestCase):

    def test_mobilenetv2_url(self):
        model = nbox.load("torchvision/mobilenetv2", pretrained = True)
        model.eval()
        image = utils.get_image("https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg")
        out = model(image)
        pred = out[0].topk(5).indices.tolist()
        self.assertEqual(pred, URL_CAT_TARGET_LIST)

    def test_mobilenetv2_filepath(self):
        model = nbox.load("torchvision/mobilenetv2", pretrained = True)
        model.eval()
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model(image)
        pred = out[0].topk(5).indices.tolist()
        self.assertEqual(pred, ASSET_CAT_TARGET_LIST)

    def test_mobilenetv2_numpy(self):
        import numpy as np
        model = nbox.load("torchvision/mobilenetv2", pretrained = True)
        image = np.random.randint(low = 0, high = 256, size = (224, 224, 3)).astype(np.uint8)
        out = model(image)

    def test_mobilenetv2_url_batch(self):
        model = nbox.load("torchvision/mobilenetv2", pretrained = True)
        image = utils.get_image("https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg")
        out = model([image, image])

    def test_mobilenetv2_filepath_batch(self):
        import os
        model = nbox.load("torchvision/mobilenetv2", pretrained = True)
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model([image, image])

    def test_mobilenetv2_numpy_batch(self):
        import numpy as np
        model = nbox.load("torchvision/mobilenetv2", pretrained = True)
        image = np.random.randint(low = 0, high = 256, size = (224, 224, 3)).astype(np.uint8)
        out = model([image, image])

if __name__ == '__main__':
    unittest.main()
