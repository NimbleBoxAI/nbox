# Nbox

A library that makes using a host of models provided by the opensource community a lot more easier. 

> The entire purpose of this package is to make inference chill.

## Installation

The library is pip installable so just run this and you should be good to go.

```bash
pip install nbox
```

## Usage

```python
import nbox

# As all these models come from the popular frameworks you use on daily basis
# such as torchvision or efficient_pytorch they have same parameters you can pass to the load function 
model = nbox.load("mobilenetv2", pretrained=False) # pretrained=TrueIf you want to use the pre trained version

# nbox makes inference the priority so you can
out = model('./tests/assets/cat.jpg') # pass it image path
out = model(Image.open('./tests/assets/cat.jpg')) # pass it PIL.Image
out = model(np.array(Image.open('./tests/assets/cat.jpg'))) # pass it numpy arrays
out = model(['./tests/assets/cat.jpg', './tests/assets/cat.jpg']) # pass it a list for batch inference

# To access the underlying model, it could be pytorch or tensorflow depending on the model.
underlying_model = model.get_model()
```

## Things for Repo

- Using [`poetry`](https://python-poetry.org/) for proper package management as @cshubhamrao says.
  - Add new packages with `poetry add <name>`. Do not add `torch`, `tensorflow` and others, useless burden to manage those.
  - When pushing to pypi just do `poetry build && poetry publish` this manages all the things around
- Install `pytest` and then run `pytest tests/ -v`.
- Using `black` for formatting, VSCode to the moon.
