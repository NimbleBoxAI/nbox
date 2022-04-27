<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>

# 🏖️ Nbox

`nbox` is NimbleBox.ai's official SDK.

> The entire purpose of this package is to make using ML 🥶.

```
pip install nbox
```

## 🔥 Usage

`nbox` provides first class support API for all NimbleBox.ai infrastructure (NBX-Build, Jobs, Deploy) and services (NBX-Workspaces) components. Write jobs using `nbox.Operators`:

```python
from nbox import Operator
from nbox.nbxlib.ops import Magic

# define a class object
weekly_trainer: Operator = Magic()

# call your operators
weekly_trainer(
  pass_values = "directly",
)

# confident? deploy it to your cloud
weekly_trainer.deploy(
  job_id_or_name = "magic_jobs",
  schedule = Schedule(4, 30, ['fri']) # schedule like humans
)
```

Deploy your machine learning or statistical models:

```python
from nbox import Model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# define your pre and post processing functions
def pre(x: Dict):
  return AutoTokenizer(**x)

# load your classifier with functions
model = AutoModelForSequenceClassification.from_pretrained("distill-bert")
classifier = Model(model, pre = pre)

# call your model
classifier(f"Is this a good picture?")

# get full control on exporting it
spec = classifier.torch_to_onnx(
  TorchToOnnx(...)
)

# confident? deploy it your cloud
url, key = classifier.deploy(
  spec, deployment_id_or_name = "classification"
)

# use it anywhere
pred = requests.post(
  url,
  json = {
    "text": f"Is this a good picture?"
  },
  header = {"Authorization": f"Bearer {key}"}
).json()
```

# 🧩 License

The code in thist repo is licensed as [Apache License 2.0](./LICENSE). Please check for individual repositories for licenses.
