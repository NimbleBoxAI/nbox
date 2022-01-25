<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>

# 🏖️ Nbox

`master` code is working, pypi `nbox` is breaking.

A library that makes using a host of models provided by the opensource community a lot more easier. 

> The entire purpose of this package is to make using models 🥶.

```
pip install nbox
```

#### Current LoC

```
SLOC	Directory	SLOC-by-Language (Sorted)
996     top_dir         python=996
88      framework       python=88

Totals grouped by language (dominant language first):
python:        1084 (100.00%)
```

## 🔥 Usage

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

## ⚙️ CLI

Just add this to your dockerfile or github actions.

```
NBX_AUTH=1 python -m nbox deploy --model_path=my/model.onnx --deployment_type="nbox"

# or for more details

python -m nbox --help
```

## ✏️ Things for Repo

- Using [`poetry`](https://python-poetry.org/) for proper package management as @cshubhamrao says.
  - Add new packages with `poetry add <name>`. Do not add `torch`, `tensorflow` and others, useless burden to manage those.
  - When pushing to pypi just do `poetry build && poetry publish` this manages all the things around
- Install `pytest` and then run `pytest tests/ -v`.
- Using `black` for formatting, VSCode to the moon.
- To make the docs:
  ```
  # from current folder
  sphinx-apidoc -o docs/source/ ./nbox/ -M -e
  cd docs && make html
  cd ../build/html && python3 -m http.server 80
  ```

# 🧩 License

The code in thist repo is licensed as [BSD 3-Clause](./LICENSE). Please check for individual repositories for licenses.
