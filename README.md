<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>
[![PyPI - Python
Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://pypi.org/project/nbox/)
[![Downloads](https://pepy.tech/badge/nbox)](https://pepy.tech/project/nbox)
![GitHub](https://img.shields.io/badge/license-Apache--2.0-blueviolet)

## π§ What is Nbox?

`nbox` provides first class support API for all NimbleBox.ai infrastructure (NBX-Build, Jobs, Deploy) and services (NBX-Workspaces) components. Write jobs using `nbox.Operators`

# π€·Why nimblebox

- Write and execute code in Python
- Document your code that supports mathematical equations
- Create/Upload/Share notebooks
- Import notebooks from your local machine
- Import/Publish notebooks from/to GitHub
- Import external datasets (e.g. from Kaggle)
- Integrate PyTorch, TensorFlow, Keras, OpenCV
- Share your projects
- Collaborate with your team

# π Features

### ποΈ Freedom To Build
![Build Landing Page (2)](https://user-images.githubusercontent.com/89596037/181773716-ba63f167-af0d-48aa-921a-02e13238c0f2.gif)


### π¦Ύ Automate with Ease
![Jobs Landing Page](https://user-images.githubusercontent.com/89596037/181774553-99120354-72f5-4064-9216-4f8a5aa050be.gif)



### π Intuitive Dashboard
![Deploy Landing Page](https://user-images.githubusercontent.com/89596037/181775468-cc342a30-d87e-4576-8bdd-8ffdd75ff759.gif)


# π Get Started


**Install the package from pipy:**

```pip install nbox```


For convinience you should add nbox to your path by setting up an alias. Throughout the rest of the documentation we will be using nbx as the CLI:

```# go to your .bashrc or .zshrc and add
alias nbx="python3 -m nbox"
```



When loading nbox for the first time, it will prompt you the username and password and create a secrets file at ```~/.nbx/secrets.json. ``` This file then contains all the information that you donβt have to fetch manually again.


## APIs

Our APIs are deep, user functions are kept to minimum and most relavant. This documentation contains the full spec of everything, but hereβs all the APIs you need to know:

```
nbox
βββ Model          # Framework agnostic Model
β   βββ __call__
β   βββ deploy
β   βββ train_on_instance (WIP)
β   βββ train_on_jobs (WIP)
βββ Operators      # How jobs are combinations of operators
β   βββ __call__
β   βββ deploy
βββ Jobs           # For controlling all your jobs
β   βββ logs       # stream logs right on your terminal
β   βββ trigger    # manually trigger a job
βββ Instance
   βββ __call__    # Run any command on the instance
   βββ mv (WIP)    # Move files to and from NBX-Build
```
### Deploy your machine learning or statistical models:

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
# π How to get help?

Join our [discord](https://discord.gg/qYZHxMaCsE) and someone from our community or engineering team will respond!

## πRead our [Blog](https://nimblebox.ai/blog).


# π§© License

The code in thist repo is licensed as [Apache License 2.0](./LICENSE). Please check for individual repositories for licenses.
