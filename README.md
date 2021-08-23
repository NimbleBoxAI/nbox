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

# nbox makes inference the priority so you can
out = model('cat.jpg')                       # pass it image path
out = model(Image.open('cat.jpg'))           # pass it PIL.Image
out = model(np.array(Image.open('cat.jpg'))) # pass it numpy arrays
out = model(['cat.jpg', 'cat.jpg'])          # pass it a list for batch inference

# To access the underlying framework dependent model
underlying_model = model.get_model()
```

## Things for Repo

- Using [`poetry`](https://python-poetry.org/) for proper package management as @cshubhamrao says.
  - Add new packages with `poetry add <name>`. Do not add `torch`, `tensorflow` and others, useless burden to manage those.
  - When pushing to pypi just do `poetry build && poetry publish` this manages all the things around
- Install `pytest` and then run `pytest tests/ -v`.
- Using `black` for formatting, VSCode to the moon.

# License

The code in thist repo is licensed as [BSD 3-Clause](./LICENSE). Please check for individual repositories for licenses. Here are some of them:

- [`torch`](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- [`transformers`](https://github.com/huggingface/transformers/blob/master/LICENSE)
- [`efficientnet-pytorch`](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE)
