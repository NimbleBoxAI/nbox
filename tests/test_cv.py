import os
import unittest
from functools import lru_cache

import nbox
from nbox import utils

# we check forward pass works and that results are reproducible
URL_CAT_TARGET_LIST = [78, 434, 700, 419, 622]
ASSET_CAT_TARGET_LIST = [111, 78, 845, 626, 418]

@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


class ImportComputerVision(unittest.TestCase):
    def test_mobilenetv2_url(self):
        model = get_model("torchvision/mobilenetv2", pretrained=True)
        model.eval()
        image = utils.get_image(
            "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        )
        out = model(image)
        pred = out[0].topk(5).indices.tolist()
        self.assertEqual(pred, URL_CAT_TARGET_LIST)

    def test_mobilenetv2_filepath(self):
        model = get_model("torchvision/mobilenetv2", pretrained=True)
        model.eval()
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model(image)
        pred = out[0].topk(5).indices.tolist()
        self.assertEqual(pred, ASSET_CAT_TARGET_LIST)

    def test_mobilenetv2_numpy(self):
        import numpy as np

        model = get_model("torchvision/mobilenetv2", pretrained=True)
        image = np.random.randint(low=0, high=256, size=(224, 224, 3)).astype(np.uint8)
        out = model(image)

    def test_mobilenetv2_url_batch(self):
        model = get_model("torchvision/mobilenetv2", pretrained=True)
        image = utils.get_image(
            "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        )
        out = model([image, image])

    def test_mobilenetv2_filepath_batch(self):
        import os

        model = get_model("torchvision/mobilenetv2", pretrained=True)
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model([image, image])

    def test_mobilenetv2_numpy_batch(self):
        import numpy as np

        model = get_model("torchvision/mobilenetv2", pretrained=True)
        image = np.random.randint(low=0, high=256, size=(224, 224, 3)).astype(np.uint8)
        out = model([image, image])


class ParserTest(unittest.TestCase):

    def test_url(self):
        parser = nbox.model.ImageParser()
        out = parser("https://i.guim.co.uk/img/media/6088d89032f8673c3473567a91157080840a7bb8/413_955_2808_1685/master/2808.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=412cc526a799b2d3fff991129cb8f030")
        self.assertEqual(list(out.shape), [1, 3, 1200, 1200])

    def test_filepath(self):
        parser = nbox.model.ImageParser()
        path = utils.join(utils.folder(__file__), "assets/cat.jpg")
        out = parser(path)
        self.assertEqual(list(out.shape), [1, 3, 720, 1280])

        out = parser(["./tests/assets/cat.jpg", "./tests/assets/cat.jpg"])
        self.assertEqual(list(out.shape), [2, 3, 720, 1280])

    def test_torch(self):
        import torch

        parser = nbox.model.ImageParser()
        out = parser(torch.randn(3, 224, 224))
        self.assertEqual(list(out.shape), [1, 3, 224, 224])

    def test_torch_list(self):
        import torch

        parser = nbox.model.ImageParser()
        out = parser([torch.randn(3, 224, 224), torch.randn(3, 224, 224)])
        self.assertEqual(list(out.shape), [2, 3, 224, 224])

    def test_numpy(self):
        import numpy as np

        parser = nbox.model.ImageParser()
        out = parser(np.random.randint(low=0, high=256, size=(224, 224, 3)))
        self.assertEqual(list(out.shape), [1, 3, 224, 224])

    def test_numpy_list(self):
        import numpy as np

        parser = nbox.model.ImageParser()
        out = parser([np.random.randint(low=0, high=256, size=(224, 224, 3)), np.random.randint(low=0, high=256, size=(224, 224, 3))])
        self.assertEqual(list(out.shape), [2, 3, 224, 224])

    def test_dicts(self):
        parser = nbox.model.ImageParser()
        out = parser({
            "image_0": "./tests/assets/cat.jpg",
            "image_1": "https://i.guim.co.uk/img/media/6088d89032f8673c3473567a91157080840a7bb8/413_955_2808_1685/master/2808.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=412cc526a799b2d3fff991129cb8f030"
        })
        self.assertEqual(
            {
                "image_0": list(out["image_0"].shape),
                "image_1": list(out["image_1"].shape)
            },
            {
                "image_0": [1, 3, 720, 1280],
                "image_1": [1, 3, 1200, 1200]
            }
        )

    def test_listed_dict(self):
        parser = nbox.model.ImageParser()
        out = parser([
            {
                "image_0": "./tests/assets/cat.jpg",
                "image_1": "https://i.guim.co.uk/img/media/6088d89032f8673c3473567a91157080840a7bb8/413_955_2808_1685/master/2808.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=412cc526a799b2d3fff991129cb8f030"
            },
            {
                "image_0": "./tests/assets/cat.jpg",
                "image_1": "https://i.guim.co.uk/img/media/6088d89032f8673c3473567a91157080840a7bb8/413_955_2808_1685/master/2808.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=412cc526a799b2d3fff991129cb8f030"
            }
        ])
        self.assertEqual(
            {
                "image_0": list(out["image_0"].shape),
                "image_1": list(out["image_1"].shape)
            },
            {
                "image_0": [2, 3, 720, 1280],
                "image_1": [2, 3, 1200, 1200]
            }
        )
    
    def test_dict_unequal_items(self):
        import torch

        parser = nbox.model.ImageParser()
        out = parser({
            "image_list": [
                "./tests/assets/cat.jpg",
                "./tests/assets/cat.jpg"
            ],
            "image_tensor": torch.randn(3, 224, 224),
            "image_tensor_list": [
                torch.randn(3, 224, 224),
                torch.randn(3, 224, 224),
                torch.randn(3, 224, 224),
            ]
        })

        self.assertEqual(
            {
                "image_list": list(out["image_list"].shape),
                "image_tensor": list(out["image_tensor"].shape),
                "image_tensor_list": list(out["image_tensor_list"].shape)
            },
            {
                "image_list": [2, 3, 720, 1280],
                "image_tensor": [1, 3, 224, 224],
                "image_tensor_list": [3, 3, 224, 224]
            }
        )

    # these tests should fail
    @unittest.expectedFailure
    def test_listed_dict_unequal_keys(self):
        parser = nbox.model.ImageParser()
        out = parser([
            {
                "image_0": "./tests/assets/cat.jpg",
                "image_1": "https://i.guim.co.uk/img/media/6088d89032f8673c3473567a91157080840a7bb8/413_955_2808_1685/master/2808.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=412cc526a799b2d3fff991129cb8f030"
            },
            {
                "image_0": "./tests/assets/cat.jpg",
                "image_2": "https://i.guim.co.uk/img/media/6088d89032f8673c3473567a91157080840a7bb8/413_955_2808_1685/master/2808.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=412cc526a799b2d3fff991129cb8f030"
            }
        ])
        
if __name__ == "__main__":
    unittest.main()
