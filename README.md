<a href="https://nimblebox.ai/" target="_blank"><img src="./assets/built_at_nbx.svg" align="right"></a>
[![PyPI - Python
Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://pypi.org/project/nbox/)
[![Downloads](https://pepy.tech/badge/nbox)](https://pepy.tech/project/nbox)
![GitHub](https://img.shields.io/badge/license-Apache--2.0-blueviolet)

## 🧐 What is Nbox?

`nbox` provides first class API support for all NimbleBox.ai infrastructure (NBX-Build, Jobs, Deploy) and services (NBX-Workspaces) components. Write jobs using `nbox.Operators`

# 🤷Why NimbleBox

- Write and execute code in Python
- Document your code that supports mathematical equations
- Create/Upload/Share notebooks
- Import notebooks from your local machine
- Import/Publish notebooks from/to GitHub
- Import external datasets (e.g. from Kaggle)
- Integrate PyTorch, TensorFlow, Keras, OpenCV
- Share your projects
- Collaborate with your team

# 🚀 Startup Program

<a href="https://nimblebox.ai/startup-program"><img width="1281" alt="image" src="https://user-images.githubusercontent.com/89596037/188064820-c372f895-fef1-4a84-bd95-8a9a5c3d13d1.png">


#### If you're a new startup with(<$1M raised,<3 years since founded) then you're in luck to be the part of our startup program!
#### Get *$420k* worth of deals on your favorite tools 
##### <a href="https://nimblebox.ai/startup-program">Check it out !
# 🎚 Features

### 🏗️ Freedom To Build
![Build Landing Page (2)](https://user-images.githubusercontent.com/89596037/181773716-ba63f167-af0d-48aa-921a-02e13238c0f2.gif)


### 🦾 Automate with Ease
![Jobs Landing Page](https://user-images.githubusercontent.com/89596037/181774553-99120354-72f5-4064-9216-4f8a5aa050be.gif)



### 🚀 Intuitive Dashboard
![Deploy Landing Page](https://user-images.githubusercontent.com/89596037/181775468-cc342a30-d87e-4576-8bdd-8ffdd75ff759.gif)


# 🏁 Get Started


**Install the package from pypi:**


When loading nbox for the first time, it will prompt you the username and password and create a secrets file at ```~/.nbx/secrets.json. ``` This file then contains all the information that you don’t have to fetch manually again.


## APIs

Our APIs are deep, user functions are kept to minimum and most relavant. This documentation contains the full spec of everything, but here’s all the APIs you need to know:

```
nbox
├── Model          # Framework agnostic Model
│   ├── __call__
│   ├── deploy
│   ├── train_on_instance (WIP)
│   └── train_on_jobs (WIP)
├── Operators      # How jobs are combinations of operators
│   ├── __call__
│   └── deploy
├── Jobs           # For controlling all your jobs
│   ├── logs       # stream logs right on your terminal
│   └── trigger    # manually trigger a job
└── Instance
   ├── __call__    # Run any command on the instance
   └── mv (WIP)    # Move files to and from NBX-Build
```

### Deploy and run any model

Let's take this script as an example

```python
from nbox import operator, Operator
from nbox.lib.shell import ShellCommand

# define your function and wrap it as an operator
@operator()
def foo(x: Dict):
  return "bar"

# or use OO to deploy an API
@operator()
class Baz():
  def __init__(self, power: int = 2):
    # load any model that you want
    self.model = load_tf_pt_model()
    self.power = power
  
  def forward(self, x: float = 1.0):
    return {"pred": x ** self.power}    
```

Through your CLI:

```bash
# to deploy a job
nbx jobs upload file:foo 'my_first_job'

# to deploy an API
nbx serve upload file:Baz 'my_first_api'
```

# 🛟 How to get help?

Join our [discord](https://discord.gg/qYZHxMaCsE) and someone from our community or engineering team will respond!

## 🔖Read our [Blog](https://nimblebox.ai/blog).


# How to get help?

Join our [discord](https://discord.gg/qYZHxMaCsE) and someone from our community or engineering team will respond!

## 🔖Read our [Blog](https://nimblebox.ai/blog).


# 🧩 License

The code in thist repo is licensed as [Apache License 2.0](./LICENSE). Please check for individual repositories for licenses.
