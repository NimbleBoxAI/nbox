# this file has the code for nbox.Model that is the holy grail of the project

from multiprocessing.sharedctypes import Value
import os
import json
from random import random
import tarfile
from glob import glob
from copy import deepcopy
from datetime import datetime
from tempfile import gettempdir
from time import sleep

from . import utils
from .utils import logger
from .framework import get_model_mixin
from .network import deploy_model
from .framework.on_ml import ModelOutput, ModelSpec
from .jobs import Instance

# model/

class Model:
  def __init__(self, m0, m1, verbose=False):
    """Top of the stack Model class.

    Args:
      model_or_model_url (Any): Model to be wrapped or model url
      nbx_api_key (str, optional): API key for this deployed model
      category (str, optional): Input categories for each input type to the model
      tokenizer ([transformers.PreTrainedTokenizer], optional): If this is a text model then tokenizer for this. Defaults to None.
      model_key (str, optional): With what key is this model initialised, useful for public models. Defaults to None.
      model_meta (dict, optional): Extra metadata when starting the model. Defaults to None.
      verbose (bool, optional): If true provides detailed prints. Defaults to False.

    Raises:
      ValueError: If any category is "text" and tokenizer is not provided.

    This class can process the following types of models:

    .. code-block::

      ``torch.nn.Module`` |           |           |
              ``sklearn`` | __init__ > serialise > S-*
           ``nbx-deploy`` |___________|___________|____
               ``S-onnx`` |              |
        ``S-torchscript`` | .deserialise > __init__
                ``S-pkl`` |______________|___________

    Serialised models are 1-3 models that have been serialised to load later. This is especially useful for
    ``nbox-serving``, one of the server types we use in NBX Deploy, yes production. This is also part of
    YoCo. Since there is a decidate serialise function, we should have one for deserialisation as well. Use
    ``nbox.Model.desirialise`` to load a serialised model.
    """


    self.user_model = m0
    self.model_support = m1
    self.verbose = verbose
    self.model = get_model_mixin(self.user_model, self.model_support)

################################################################################
# Functions here utility functions
################################################################################

  def __repr__(self):
    return f"<nbox.Model: {self.model} >"

  def eval(self):
    """if underlying model has eval method, call it"""
    if hasattr(self.model, "eval"):
      self.model.eval()

  def train(self):
    """if underlying model has train method, call it"""
    if hasattr(self.model, "train"):
      self.model.train()

################################################################################
# Functions here are the services that NBX provides to the user and no longer
# >v0.8.7 the implementation of processing logic.
################################################################################

  def __call__(self, input_object) -> ModelOutput:
    r"""Caller is the most important UI/UX. The ``input_object`` can be anything from
    a tensor, an image file, filepath as string, string and must be processed automatically by a
    well written ``nbox.parser.BaseParser`` object . This ``__call__`` should understand the different
    usecases and manage accordingly.

    The current idea is that what ever the input, based on the category (image, text, audio, smell)
    it will be parsed through dedicated parsers that can make ingest anything.

    The entire purpose of this package is to make inference chill.

    Args:
      input_object (Any): input to be processed

    Returns:
      Any: currently this is output from the model, so if it is tensors and return dicts.
    """
    return self.model.forward(input_object)

  def serialise(
    self,
    input_object,
    model_name,
    export_type="onnx",
    return_meta = False,
    *,
    _do_tar = True,
    _unit_test = False,
    **kwargs
  ) -> str:
    """This creates a singular .nbox file that contains the model binary and config file in ``self.cache_dir``:

    Creates a folder at ``/tmp/{model_name}`` and then let's the underlying framework to fill it with anything
    that it wants. Once the entire creation of file is completed it will zip them all in ``/tmp/{model_name}.nbox``
    in process deleting all the files.

    Args:
      input_object (Any): input to be processed
      model_name (str, optional): name of the model
      export_type (str, optional): [description]. Defaults to "onnx".
      generate_ov_args (bool, optional): Return the CLI args to be passed for Intel OpenVino

    Returns:
      path (str): path for the tar file
    """

    # create the export folder
    folder = utils.join(
      utils.NBOX_HOME_DIR, f"{model_name}", datetime.now().utcnow().strftime("UTC_%Y-%m-%dT%H:%M:%S")
    ) if not _unit_test else utils.join(
      gettempdir(), f"{model_name}"
    )
    logger.debug(f"Serialising '{model_name}' to '{export_type}' at '{folder}'")
    os.makedirs(folder, exist_ok=True)
    
    # export the model
    nbox_meta = self.model.export(
      format = export_type,
      input_object = input_object,
      export_model_path = folder,
      **kwargs
    )

    meta_path = utils.join(folder, f"nbox_config.json")
    with open(meta_path, "w") as f:
      f.write(json.dumps(nbox_meta.get_dict()))

    nbx_path = os.path.join(folder, f"{model_name}.nbox")
    all_files = utils.get_files_in_folder(folder)

    if not _do_tar:
      return (all_files, nbx_path)

    with tarfile.open(nbx_path, "w|gz") as tar:
      for path in all_files:
        tar.add(path, arcname = os.path.basename(path))
        # os.remove(path)
        logger.debug(f"Removed {path}")

    return nbx_path if not return_meta else (nbx_path, nbox_meta)

  @classmethod
  def deserialise(cls, filepath):
    """This is the counterpart of ``serialise``, it deserialises the model from the .nbox file.

    Args:
      nbx_path (str): path to the .nbox file
    """
    if not tarfile.is_tarfile(filepath) or not filepath.endswith(".nbox"):
      raise ValueError(f"{filepath} is not a valid .nbox file")

    with tarfile.open(filepath, "r:gz") as tar:
      folder = utils.join(gettempdir(), os.path.basename(filepath).replace(".nbox", ""))
      logger.debug(f"Extracted to folder: {folder}")
      tar.extractall(folder)

    # go into the currect folder so it's much easier to load things
    _pre_dir = deepcopy(os.getcwd())
    os.chdir(folder)

    with open("./nbox_config.json", "r") as f:
      nbox_meta = json.load(f)
    m0, m1 = get_model_mixin(ModelSpec(**nbox_meta), deserialise=True)
    
    os.chdir(_pre_dir) # shift path back to original
    return cls(m0, m1)

  @staticmethod
  def train_on_instance(
    instance: Instance,
    serialised_fn: callable,
    train_fn: callable,
    other_args: tuple = (), # any other arguments to be passed to the train_fn
    target_folder: str = "/", # anything after /project folder
    shutdown_once_done: bool = False,
    *,
    _unit_test = False,
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

    from types import SimpleNamespace
    train_fn_path = utils.join(utils.folder(nbx_path), "train_fn.dill")
    logger.debug(f"Train function saved at {train_fn_path}")
    utils.to_pickle(SimpleNamespace(train_fn = train_fn, args = other_args), train_fn_path)
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
      utils.join(utils.folder(__file__), "assets", "train_fn.jina"),
      utils.join(run_folder, "run.py")
    )

    pid = instance(utils.join(run_folder, "run.py"))
    instance.stream_logs(pid)

    if shutdown_once_done:
      instance.stop()

  def deploy(
    self,
    input_object,
    model_name,
    export_type="onnx",
    wait_for_deployment=False,
    deployment_id=None,
    deployment_name=None,
    **ser_kwargs,
  ):
    """NBX-Deploy `read more <https://nimbleboxai.github.io/nbox/nbox.model.html>`_

    This deploys the current model onto our managed K8s clusters. This tight product service integration
    is very crucial for us and is the best way to make deploy a model for usage.

    Raises appropriate assertion errors for strict checking of inputs

    Args:
      input_object (Any): input to be processed
      model_name (str, optional): custom model name for this model
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

    export_model_path, nbox_meta = self.serialise(
      input_object = input_object,
      model_name = model_name,
      export_type = export_type,
      _unit_test = False,
      return_meta = True,
      **ser_kwargs
    )

    nbox_meta["deployment_type"] = "nbox" # TODO: @yashbonde remove hardcode
    nbox_meta["deployment_id"] = deployment_id
    nbox_meta["deployment_name"] = deployment_name

    # OCD baby!
    return deploy_model(
      export_model_path=export_model_path,
      nbox_meta=nbox_meta,
      wait_for_deployment=wait_for_deployment,
    )
