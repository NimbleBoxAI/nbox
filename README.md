# Nbox

A library that makes using a host of models provided by the opensource community a lot more easier. 

> The entire purpose of this package is to make inference chill.

```
pip install nbox
```

## Usage

```python
import nbox

# As all these models come from the popular frameworks you use such as 
# torchvision, efficient_pytorch or hf.transformers
model = nbox.load("torchvision/mobilenetv2", pretrained = True)

# nbox makes inference the priority so you can use it
# pass it a list for batch inference 
out_single = model('cat.jpg')
out = model([Image.open('cat.jpg'), np.array(Image.open('cat.jpg'))])
tuple(out.shape) == (2, 1000)

# deploy the model to a managed kubernetes cluster on NimbleBox.ai
url_endpoint, nbx_api_key = model.deploy()

# or load a cloud infer model and use seamlessly
model = nbox.load(
  model_key_or_url = url_endpoint,
  nbx_api_key = nbx_api_key,
  category = "image"
)

# Deja-Vu!
out_single = model('cat.jpg')
out = model([Image.open('cat.jpg'), np.array(Image.open('cat.jpg'))])
tuple(out.shape) == (2, 1000)
```

## Things for Repo

- Using [`poetry`](https://python-poetry.org/) for proper package management as @cshubhamrao says.
  - Add new packages with `poetry add <name>`. Do not add `torch`, `tensorflow` and others, useless burden to manage those.
  - When pushing to pypi just do `poetry build && poetry publish` this manages all the things around
- Install `pytest` and then run `pytest tests/ -v`.
- Using `black` for formatting, VSCode to the moon.

# License

The code in thist repo is licensed as [BSD 3-Clause](./LICENSE). Please check for individual repositories for licenses. Here are some of them:

### Requirements

- [`rich`](https://github.com/willmcgugan/rich/blob/master/LICENSE)

### Model Sources

99% of the credit for `nbox` goes to the amazing people behind these projects:

- [`torch`](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- [`transformers`](https://github.com/huggingface/transformers/blob/master/LICENSE)
- [`efficientnet-pytorch`](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE)
