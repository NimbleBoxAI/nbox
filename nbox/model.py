"""
This file has the code to run ``nbox.Model`` which is top of the stack class. It can load
any arbitrary model and run it, deploy it (if possible to export), train it (WIP). It has gone
through it's fare share of refactoring and is now much simpler, in terms of functions while
providing very abstract value.

tl;dr
-----

Underneath this system uses a simple and effective strategy. All the supported methods for
all the frameworks (tensoflow, pytorch, etc) are in a registery, whenever the user sends
any model, it is matched and all functions for that framework (eg. ``sklearn_to_pickle``)
are loaded. This then provides first class support eg. ``delay_forecaster.torch_to_onnx()``,
combined with the protobuf style message stub system this provides all configurations to
user and leaves us incharge of simply executing it.

Story Mode
----------

Let's just say your organisation is working on project for predicting "User Fraud Prevention",
and you already have a working baseline in scikit learn. You have already trained the model
and now want to deploy it to production.

.. code-block:: python

  # get your baseline model
  sklearn_model: 'SklearnModel' = get_skl_model(...)
  fraud_scoring: Model = Model(model = sklearn_model, method = "predict_proba")

  # sklearn models consume numpy tesors so we send them those
  full_matrix: np.ndarray = get_test_data()
  y_pred: np.ndarray = stock_prediction(x = full_matrix[:, :-1])

Before deploying you can write simple functions that can be used to preprocess and postprocess
the incoming data. Your function consumes``UserProto`` or a ``Dict`` that can be parsed to
proto. So you create a new model like this:

.. code-block:: python

  def _pre(user: Union[UserProto, Dict]) -> Dict:
    if isinstance(str):
      user = dict_to_message(user, UserProto) # deployment works on REST so load dict to proto

    # maybe you want to update your DB
    sql.execute(f'INSERT INTO users (id, name) VALUES ({user.id}, {user.name})')

    ... # your business logic here

    return {"X": user.feature_vector.numpy(),} # NOTE: returns dict

  fraud_scoring = Model(
    model = sklearn_model,
    method = "predict_proba",
    pre = _pre
  )

  # You can now pass both proto and dict to the model
  user: UserProto = get_user_from_db()
  prediction: Dict = fraud_scoring(user)

If you want to add a post processing (eg. here I am just converting the prediction to a list):

.. code-block:: python

  def _post(pred: Dict) -> Dict:
    # maybe you want to update your DB
    sql.execute(f'UPDATE users SET fraud_score = {pred["fraud_score"]} WHERE id = {pred["user_id"]}')

    ... # your business logic here

    return {k:v.tolist() for k,v in pred} # NOTE: returns json serializable dict

  fraud_scoring = Model(
    model = sklearn_model,
    method = "predict_proba",
    pre = _pre,
    post = _post
  )

When you want to deploy the model, you will need to export it to some format. You don't have to
search through documentation to find out supported models, they are already in the object as
``extra_fns`` attribute (``fraud_scoring.extra_fns``). You chose to go the reliable way of pickling
and unpickling.

.. code-block:: python

  from nbox.framework import SklearnToPickle, NboxOptions, ModelSpec

  # get a protobuf dataclass for the spec
  model_spec: ModelSpec = fraud_scoring.sklearn_to_pickle(
    SklearnToPickle( ... ), # fill values based on your preferance
    NboxOptions( ... ) # fill values based on your preferance
  )

You have the ``ModelSpec`` object and you can now deploy it. All the information is packaged
in a nice serving that will be put on K8s.

.. code-block:: python

  url, key = fraud_scoring.deploy(
    model_spec = model_spec,
    ... # other deployment args
  )

Here's documentation to the code:
"""

import os
import tarfile
from typing import Any, Dict, Union
from types import SimpleNamespace


from . import utils as U
from .utils import logger
from .init import nbox_ws_v1
from .network import deploy_serving
from .instance import Instance
from .subway import NboxModelSubway
from .framework import get_model_functions
from .framework import ModelSpec, Deployment, NboxOptions
from .messages import message_to_json, dict_to_message

# model/

class Model:
  def __init__(
    self,
    model: Any,
    method: str = None,
    pre: callable = None,
    post: callable = None,
    model_spec: ModelSpec = None,
    verbose: bool = False
  ):
    """Top of the stack Model class.

    Args:
      model (Any): First input to the model
      method (str, optional): If there is a specific function to call for the model.
        if ``None`` then ``__call__`` will be made else ``model.method`` will be called.
        Defaults to None.
      pre (callable): preprocessing function to be applied to the ``input_object``
      post (callable): postprocessing function to be applied to the model prediction
      model_spec (ModelSpec, optional): ModelSpec object to be used for the model.
      verbose (bool, optional): If true provides detailed prints. Defaults to False.
    """

    if model == NboxModelSubway:
      self.model = NboxModelSubway(model)
      method = None
    else:
      self.model = model
    self.method = method
    self.forward_fn = self.model if method == None else getattr(self.model, method)
    self.pre = pre if pre != None else lambda x: x
    self.post = post if post != None else lambda x: x
    self.model_spec = model_spec
    self.verbose = verbose

    self.extra_fns: Dict = get_model_functions(self.model)
    for (fn_name, fn_meta) in self.extra_fns.items():
      fn, _ = fn_meta
      setattr(self, fn_name, fn)
      logger.debug(f"Adding {fn_name}")

  ################################################################################
  # Utility functions
  ################################################################################

  def __repr__(self):
    return f"<nbox.Model: {self.model} >"

  def __dir__(self):
    # be careful: https://docs.python.org/3/library/functions.html#dir
    return [
      # core functions
      "__init__",
      "__repr__",
      "__call__",
      "deserialise",
      "deploy",
      "train_on_instance",

      # attributes
      "model",
      "method",
      "forward_fn",
      "pre",
      "post",
      "model_spec",
      "verbose",
      "extra_fns",

      # framework related functions
      *tuple(self.extra_fns.keys())
    ]

  ################################################################################
  # Functions here are the services that NBX provides
  ################################################################################

  def __call__(self, input_object) -> Any:
    r"""Call is the most important UI/UX. The ``input_object`` can be anything from
    a tensor, an image file, filepath as string, string and is processed by ``pre`` function.

    The entire purpose of this package is to make ML chill.

    Args:
      input_object (Any): input to be processed
    """
    pre_out = self.pre(input_object) # pre processing output
    model_out = self.forward_fn(**pre_out) # model prediction
    post_out = self.post(model_out) # post processing output
    return post_out

  @classmethod
  def deserialise(cls, model_spec: Union[ModelSpec, Dict], folder) -> 'Model':
    """Load ``ModelSpec`` and ``folder`` with the files in it and return a ``Model`` object.
    
    Args:
      model_spec (Union[ModelSpec, Dict]): ModelSpec object or dictionary of the model_spec
      folder (str): folder where the model files are stored
    """
    if isinstance(model_spec, dict):
      _model_spec = ModelSpec()
      model_spec = dict_to_message(model_spec, _model_spec)
    logger.info(f"{model_spec}")
    init_data = U.from_pickle(U.join(folder, "model.extras.pkl"))

    # now need to load the model from the serialised object
    _class, _method = model_spec.target.method.split(".")
    exec(f"from .framework.ml import {_class}")
    _module = eval(f"{_class}")
    loader, options_cls = _module._METHODS.get(_method)
    m0 = loader(
      user_options = options_cls(),
      nbox_options = NboxOptions(model_name = model_spec.name, folder = folder, create_folder = False)
    )
    return cls(m0, method = init_data["method"], pre = init_data["pre"], post = init_data["post"], model_spec = model_spec)

  def deploy(
    self,
    model_spec: ModelSpec,
    deployment_id_or_name: str = None,
    workspace_id: str = None,
    wait_for_deployment=True,
    *,
    _unittest = False
  ):
    """Serve your model on NBX-Deploy `read more <https://nimbleboxai.github.io/nbox/nbox.model.html>`_

    Args:
      model_spec (nbox.framework.ModelSpec): ModelSpec object
      deployment_id_or_name (str, optional): Deployment information through ID or name, if not
        provided will create a new deployment group with the given name
      workspace_id (str, optional): Workspace ID to deploy the model to. If not provided
        will use the personal workspace.
      wait_for_deployment (bool, optional): Block thread till deployment to be ready.
    """
    if workspace_id == None:
      stub_all_depl = nbox_ws_v1.user.deployments
    else:
      stub_all_depl = nbox_ws_v1.workspace.u(workspace_id).deployments
    logger.debug(f"deployments stub: {stub_all_depl}")

    _deploy_proto = Deployment(
      workspace_id = workspace_id,
      type = Deployment.DeploymentTypes.NBOX_SERVING # ignored for now
    )

    deployments = list(filter(
      lambda x: x["deployment_id"] == deployment_id_or_name or x["deployment_name"] == deployment_id_or_name,
      stub_all_depl()["data"]
    ))
    if len(deployments) == 0:
      logger.warning(f"No deployment found with id '{deployment_id_or_name}', creating one with same name")
      _deploy_proto.id = None
      _deploy_proto.name = deployment_id_or_name
    elif len(deployments) > 1:
      raise ValueError(f"Multiple deployments found for '{deployment_id_or_name}', try passing ID")
    else:
      data = deployments[0]
      _deploy_proto.id = data["deployment_id"]
      _deploy_proto.name = data["deployment_name"]

    # update model spec with deployment related information
    model_spec.deploy.CopyFrom(_deploy_proto)

    # pack everything nicely
    folder = model_spec.folder
    extras = U.join(folder, f"model.extras.pkl")
    logger.info(f"Writing model.extras: {extras}")
    U.to_pickle({"pre": self.pre, "post": self.post, "method": self.method}, extras)

    req = U.join(folder, "requirements.txt")
    with open(req, "w") as f:
      logger.info(f"Writing the requirements file: {req}")
      f.write("\n".join(model_spec.requirements))

    meta_path = U.join(folder, f"nbox_config.json")
    with open(meta_path, "w") as f:
      logger.info(f"Writing nbox.meta: {meta_path}")
      f.write(message_to_json(model_spec))

    nbx_path = U.join(folder, f"{folder}.nbox")
    all_files = U.get_files_in_folder(folder)
    with tarfile.open(nbx_path, "w|gz") as tar:
      logger.info(f"Writing: {nbx_path}")
      for path in all_files:
        tar.add(path, arcname = os.path.basename(path))
        logger.debug(f"Removed {path}")

    if _unittest:
      # returns the minimum information needed to deserialise the model
      return model_spec, folder

    # OCD baby!
    return deploy_serving(
      export_model_path=nbx_path,
      stub_all_depl=stub_all_depl,
      model_spec=model_spec,
      wait_for_deployment=wait_for_deployment,
    )

  @staticmethod
  def train_on_instance(
    instance: Instance,
    serialised_fn: callable,
    train_fn: callable,
    other_args: tuple = (), # any other arguments to be passed to the train_fn
    target_folder: str = "/", # anything after /project folder
    shutdown_once_done: bool = False,
    *,
    _unit_test: bool = False,
  ):
    """Train this model on an NBX-Build Instance. Though this function is generic enough to execute
    any arbitrary code, this is built primarily for internal use.

    EXPERIMENTAL: FEATURES MIGHT BREAK

    Args:
      instance (Instance): Instance to train the model on
      serialised_fn (callable): path to the serialised tar file
      train_fn (callable): pure function that trains the model
      other_args (Any, optional): any other arguments to be passed to the train_fn
      target_folder (str, optional): folder on the ``instance`` to run this program in,
        will run in folder ``/project/{target_folder}/``
      shutdown_once_done (bool, optional): if true, shutdown the instance once training is done.
    """

    assert instance.status == "RUNNING", f"Instance {instance.id} is not running"
    all_files, nbx_path = serialised_fn(_do_tar = False, _unit_test = _unit_test)

    train_fn_path = U.join(U.folder(nbx_path), "train_fn.dill")
    logger.debug(f"Train function saved at {train_fn_path}")
    U.to_pickle(SimpleNamespace(train_fn = train_fn, args = other_args), train_fn_path)
    all_files.append(train_fn_path)

    logger.debug(f"Creating nbox zip: {nbx_path}")
    with tarfile.open(nbx_path, "w|gz") as tar:
      for path in all_files:
        tar.add(path, arcname = os.path.basename(path))
        # os.remove(path)
        logger.debug(f"Removed {path}")

    run_folder = f"/project/{target_folder}/"

    instance.mv(nbx_path, run_folder)
    instance("cd {}; python3 -m nbox.train_fn".format(run_folder))
    instance.mv(
      U.join(U.folder(__file__), "assets", "train_fn.jinja"),
      U.join(run_folder, "run.py")
    )

    pid = instance(U.join(run_folder, "run.py"))
    instance.stream_logs(pid)

    if shutdown_once_done:
      instance.stop()

  @staticmethod
  def train_on_job():
    """Train this model on NBX-Jobs. This same experience can be given by creating a new job
    that then can then be populated how we currently create a new job using NBX-Jobs CLI
    ```
    nbx jobs new --help
    ```
    So should we really add this, is the question!
    
    EXPERIMENTAL: FEATURES MIGHT BREAK
    """
    pass


def magic_model():
  from .framework.on_ml import _get_torch_model, _get_sklearn_model, _get_default_forward
  for fn in [_get_torch_model, _get_sklearn_model, _get_default_forward]:
    try:
      model_kwargs, sample_input = fn()
    except ImportError as e:
      pass

  return Model(**model_kwargs), sample_input
