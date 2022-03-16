# this file has the code for nbox.Model that is the holy grail of the project

import os
import tarfile
from typing import Any, Dict
from types import SimpleNamespace
from google.protobuf.json_format import MessageToJson, ParseDict


from . import utils as U
from .utils import logger
from .init import nbox_ws_v1
from .network import deploy_model
from .instance import Instance
from .subway import Sub30, NboxModelSubway
from .framework import get_model_functions
from .framework import ModelSpec, Deployment, NboxOptions


# model/

class Model:
  def __init__(
    self,
    model: Any,
    method: str = None,
    pre: callable = None,
    post: callable = None,
    model_spec = None,
    verbose = False
  ):
    """Top of the stack Model class.

    Args:
      model (Any): First input to the model
      method (str, optional): If there is a specific function to call for the model.
        if ``None`` then ``__call__`` will be made else ``model.method`` will be called.
        Defaults to None.
      pre (callable): preprocessing function to be applied to the ``input_object``
      post (callable): postprocessing function to be applied to the model prediction
      verbose (bool, optional): If true provides detailed prints. Defaults to False.

    Raises:
      ValueError: If any category is "text" and tokenizer is not provided.
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
  # Functions here are the services that NBX provides to the user and no longer
  # >v0.8.8-alpha the implementation of processing logic.
  ################################################################################

  def __call__(self, input_object) -> Any:
    r"""Caller is the most important UI/UX. The ``input_object`` can be anything from
    a tensor, an image file, filepath as string, string and is processed by ``pre`` function.

    The entire purpose of this package is to make inference chill.

    Args:
      input_object (Any): input to be processed
    """
    pre_out = self.pre(input_object) # pre processing output
    model_out = self.forward_fn(**pre_out) # model prediction
    post_out = self.post(model_out) # post processing output
    return post_out

  @classmethod
  def deserialise(cls, model_spec, folder) -> 'Model':
    """load spec and files in the folder to start running the model."""
    if isinstance(model_spec, dict):
      _model_spec = ModelSpec()
      ParseDict(model_spec, _model_spec)
      model_spec = _model_spec
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
    """NBX-Deploy `read more <https://nimbleboxai.github.io/nbox/nbox.model.html>`_

    This deploys the current model onto our managed K8s clusters. This tight product service integration
    is very crucial for us and is the best way to make deploy a model for usage.

    Raises appropriate assertion errors for strict checking of inputs

    Args:
      model_spec (nbox.framework.ModelSpec): what the name says!
      export_type (str, optional): Export type for the model, check documentation for more details
      wait_for_deployment (bool, optional): wait for deployment to complete
      deployment_id (str, optional): ``deployment_id`` to put this model under, if you do not pass this
        it will automatically create a new deployment check `platform <https://nimblebox.ai/oneclick>`_
        for more info or check the logs.
      deployment_name (str, optional): if ``deployment_id`` is not given and you want to create a new
        deployment group (ie. webserver will create a new ``deployment_id``) you can tell what name you
        want, be default it will create a random name.
      **ser_kwargs (Any, optional): keyword arguments to be passed to ``serialise`` function
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
      raise ValueError(f"Multiple deployments found for '{deployment_id_or_name}'")
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
      f.write(MessageToJson(
        model_spec,
        including_default_value_fields=True,
        preserving_proto_field_name=True,
        indent=2,
        sort_keys=False,
        use_integers_for_enums=True,
        float_precision=4
      ))

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
    return deploy_model(
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
