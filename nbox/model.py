# this file has the code for nbox.Model that is the holy grail of the project

import os
import json
import shutil
import inspect
import tarfile
from glob import glob
from tempfile import gettempdir

from . import utils
from .framework import get_meta, get_model_mixin
from .network import deploy_model
from .framework.on_ml import ModelOutput

import logging
logger = logging.getLogger()


class GenericMixin:
  def eval(self):
    """if underlying model has eval method, call it"""
    if hasattr(self.model_or_model_url, "eval"):
      self.model_or_model_url.eval()

  def train(self):
    """if underlying model has train method, call it"""
    if hasattr(self.model_or_model_url, "train"):
      self.model_or_model_url.train()

  def __repr__(self):
    return f"<nbox.Model: {self.model_or_model_url} >"

# utils/

# /utils

# model/

class Model(GenericMixin):
  def __init__(
    self,
    model,
    model_support,
    cache_dir = None,
    verbose=False,

    nbx_api_key=None,
    category=None,
    tokenizer=None,
    model_key=None,
    nbox_meta=None,
    
  ):
    """Top of the stack Model class.

    >> BREAKING <<

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
    
    
    self.user_model = model
    self.model_support = model_support
    self.cache_dir = cache_dir
    self.verbose = verbose

    self.model = get_model_mixin(self.user_model, self.model_support)

    # # values coming from the blocks above
    # self.model_or_model_url = model_or_model_url
    # self.__framework = __framework
    # self.text_parser = text_parser
    # self.image_parser = image_parser
    # self.nbox_meta = nbox_meta
    # self.nbx_api_key = nbx_api_key
    # self.category = category
    # self.tokenizer = tokenizer
    # self.model_key = model_key
    # self.verbose = verbose
    # self.__device = "cpu"
    # self.cache_dir = gettempdir() if cache_dir == None else cache_dir

    logger.info(f"Model loaded successfully")
    # logger.info(f"Model framework: {self.__framework}")
    # logger.info(f"Model category: {self.category}")

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
      return_inputs (bool, optional): whether to return the inputs or not. Defaults to False.
      method(str, optional): specifically for sklearn models, this is the method to be called
        if nothing is provided then we call ``.predict()`` method.

    Returns:
      Any: currently this is output from the model, so if it is tensors and return dicts.
    """

    # the forward_pass is a method for the current model, this is responsible for parsing and
    # processing the input object.
    out = self.model.forward(input_object)

    # add post here
 
    # # convert to dictionary if needed
    # if return_dict:
    #   output_names = list(self.nbox_meta["metadata"]["outputs"].keys())
    #   if isinstance(out, (tuple, list)):
    #     out = {k: v for k, v in zip(output_names, out)}
    #   elif isinstance(out, dict):
    #     pass
    #   else:
    #     try:
    #       if isinstance(out, (torch.Tensor, np.ndarray)):
    #         out = {k: v.tolist() for k, v in zip(output_names, [out])}
    #     except NameError:
    #       pass

    #     raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")

    # if return_inputs:
    #   return out, model_input
    return out

  def _handle_input_object(self, input_object):
    """First level handling to convert the input object to a fixed object"""
    # in case of scikit learn user must ensure that the input_object is model_input
    if self.__framework == "sklearn":
      return input_object

    elif self.__framework in ["nbx", "onnx"]:
      # the beauty is that the server is using the same code as this meaning that client
      # can get away with really simple API calls
      inputs_deploy = set(self.nbox_meta["metadata"]["inputs"].keys())
      if isinstance(input_object, dict):
        inputs_client = set(input_object.keys())
        assert inputs_deploy == inputs_client, f"Inputs mismatch, deploy: {inputs_deploy}, client: {inputs_client}"
        input_object = input_object
      else:
        if len(inputs_deploy) == 1:
          input_object = {list(inputs_deploy)[0]: input_object}
        else:
          assert len(input_object) == len(inputs_deploy), f"Inputs mismatch, deploy: {inputs_deploy}, client: {len(input_object)}"
          input_object = {k: v for k, v in zip(inputs_deploy, input_object)}

    if isinstance(self.category, dict):
      assert isinstance(input_object, dict), "If category is a dict then input must be a dict"
      # check for same keys
      assert set(input_object.keys()) == set(self.category.keys())
      input_dict = {}
      for k, v in input_object.items():
        if k in self.category:
          if self.category[k] == "image":
            input_dict[k] = self.image_parser(v)
          elif self.category[k] == "text":
            input_dict[k] = self.text_parser(v)
          elif self.category[k] == "tensor":
            input_dict[k] = v
          else:
            raise ValueError(f"Unsupported category: {self.category[k]}")
      return input_dict

    elif self.category == "image":
      input_obj = self.image_parser(input_object)
      return input_obj

    elif self.category == "text":
      # perform parsing for text and pass to the model
      input_dict = self.text_parser(input_object)
      return input_dict

    # Code below this part is super flaky and is useful for sklearn model,
    # please improve this as more usecases come up
    elif self.category == None:
      if isinstance(input_object, dict):
        return {k: v.tolist() for k, v in input_object.items()}
      return input_object.tolist()

    # when user gives a list as an input, it's better just to pass it as is
    # but when the input becomes a dict, this might fail.
    return input_object

  def get_nbox_meta(self, input_object, return_kwargs = True):
    """Get the nbox meta and trace args for the model with the given input object

    Args:
      input_object (Any): input to be processed
    """
    # this function gets the nbox metadata for the the current model, based on the input_object
    if self.__framework == "nbx":
      return self.nbox_meta

    args = None
    if self.__framework == "pytorch":
      args = inspect.getfullargspec(self.model_or_model_url.forward).args
      args.remove("self")

    self.eval() # covert to eval mode
    model_output, model_input = self(input_object, return_inputs=True)

    # need to convert inputs and outputs to list / tuple
    dynamic_axes_dict = {
      0: "batch_size",
    }
    if self.category == "text":
      dynamic_axes_dict[1] = "sequence_length"

    # need to convert inputs and outputs to list / tuple
    if isinstance(model_input, dict):
      model_inputs = tuple(model_input.values())
      input_names = tuple(model_input.keys())
      input_shapes = tuple([tuple(v.shape) for k, v in model_input.items()])
    elif isinstance(model_input, (torch.Tensor, np.ndarray)):
      model_inputs = tuple([model_input])
      input_names = tuple(["input_0"]) if args is None else tuple(args)
      input_shapes = tuple([tuple(model_input.shape)])
    dynamic_axes = {i: dynamic_axes_dict for i in input_names}

    if isinstance(model_output, dict):
      output_names = tuple(model_output.keys())
      output_shapes = tuple([tuple(v.shape) for k, v in model_output.items()])
      model_output = tuple(model_output.values())
    elif isinstance(model_output, (list, tuple)):
      mo = model_output[0]
      if isinstance(mo, dict):
        # cases like [{"output_0": tensor, "output_1": tensor}]
        output_names = tuple(mo.keys())
        output_shapes = tuple([tuple(v.shape) for k, v in mo.items()])
      else:
        output_names = tuple([f"output_{i}" for i, x in enumerate(model_output)])
        output_shapes = tuple([tuple(v.shape) for v in model_output])
    elif isinstance(model_output, (torch.Tensor, np.ndarray)):
      output_names = tuple(["output_0"])
      output_shapes = (tuple(model_output.shape),)

    meta = get_meta(input_names, input_shapes, model_inputs, output_names, output_shapes, model_output)
    out = {
      "args": model_inputs,
      "outputs": model_output,
      "input_shapes": input_shapes,
      "output_shapes": output_shapes,
      "input_names": input_names,
      "output_names": output_names,
      "dynamic_axes": dynamic_axes,
    }
    if return_kwargs:
      return meta, out
    return meta

  def serialise(
    self,
    input_object,
    model_name=None,
    export_type="onnx",
    generate_ov_args = False,
    return_meta = False
  ):
    """This creates a singular .nbox file that contains the model binary and config file in ``self.cache_dir``

    Creates a folder at ``/tmp/{hash}`` and then adds three files to it:
      - ``/tmp/{hash}/{model_name}.{export_type}``
      - ``/tmp/{hash}/{model_name}.json``
      - ``/tmp/{hash}/{model_name}.nbox``

    and then later deletes the ``{model_name}.{export_type}`` and ``{model_name}.json`` and returns
    ``{model_name}.nbox`` and if required the metadata

    Args:
      input_object (Any): input to be processed
      model_name (str, optional): name of the model
      export_type (str, optional): [description]. Defaults to "onnx".
      generate_ov_args (bool, optional): Return the CLI args to be passed for Intel OpenVino
    """

    # First Step: check the args and see if conditionals are correct or not
    def __check_conditionals():
      assert self.__framework != "nbx", "This model is already deployed on the cloud"
      assert export_type in ["onnx", "torchscript", "pkl"], "Export type must be onnx, torchscript or pickle"
      if self.__framework == "sklearn":
        assert export_type in ["onnx", "pkl"], f"Export type must be onnx or pkl | got {export_type}"
      if self.__framework == "pytorch":
        assert export_type in ["onnx", "torchscript"], f"Export type must be onnx or torchscript | got {export_type}"

    # perform sanity checks on the input values
    __check_conditionals()

    # get metadata and exporing values
    nbox_meta, export_kwargs = self.get_nbox_meta(input_object, return_kwargs=True)
    _m_hash = utils.hash_(self.model_key)
    export_folder = utils.join(self.cache_dir, _m_hash)
    logger.info(f"Exporting model to {export_folder}")
    os.makedirs(export_folder, exist_ok=True)

    # intialise the console logger
    model_name = model_name if model_name is not None else f"{utils.get_random_name()}-{_m_hash[:4]}".replace("-", "_")
    export_model_path = utils.join(export_folder, f"{model_name}.{export_type}")
    logger.info("-" * 30 + f" Exporting {model_name}")

    # convert the model -> create a the spec, get the actual method for conversion
    logger.info(f"model_name: {model_name}")
    logger.info(f"Export type: {export_type}")
    
    try:
      export_fn = getattr(globals()[f"frm_{self.__framework}"], f"export_to_{export_type}", None)
      if export_fn == None:
        raise KeyError(f"Export type {export_type} not supported for {self.__framework}")
    except KeyError:
      raise ValueError(f"Framework {self.__framework} not supported (likely missing packages), check logs for more info")

    logger.info(f"Converting using: {export_fn}")
    export_fn(model=self.model_or_model_url, export_model_path=export_model_path, **export_kwargs)
    logger.info("Conversion Complete")

    # construct the output
    convert_args = None
    if generate_ov_args:
      # https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html
      input_ = ",".join(export_kwargs["input_names"])
      input_shape = ",".join([str(list(x.shape)).replace(" ", "") for x in export_kwargs["args"]])
      convert_args = f"--data_type=FP32 --input_shape={input_shape} --input={input_} "

      if self.category == "image":
        # mean and scale have to be defined for every single input
        # these values are calcaulted from uint8 -> [-1,1] -> ImageNet scaling -> uint8
        mean_values = ",".join([f"{name}[182,178,172]" for name in export_kwargs["input_names"]])
        scale_values = ",".join([f"{name}[28,27,27]" for name in export_kwargs["input_names"]])
        convert_args += f"--mean_values={mean_values} --scale_values={scale_values}"
      logger.info(convert_args)

    # write the nbox meta in a file and tarball the model and meta
    nbox_meta = {
      "metadata": nbox_meta,
      "spec": {
        "category": self.category,
        "model_key": self.model_key,
        "model_name": model_name,
        "src_framework": self.__framework,
        "export_type": export_type,
        "convert_args": convert_args
      },
    }
    meta_path = utils.join(export_folder, f"{model_name}.json")
    with open(meta_path, "w") as f:
      f.write(json.dumps(nbox_meta))

    nbx_path = os.path.join(export_folder, f"{model_name}.nbox")
    with tarfile.open(nbx_path, "w|gz") as tar:
      tar.add(export_model_path, arcname=os.path.basename(export_model_path))
      tar.add(meta_path, arcname=os.path.basename(meta_path))

    # remove the files
    os.remove(export_model_path)
    os.remove(meta_path)

    if return_meta:
      return nbx_path, nbox_meta
    return nbx_path

  @classmethod
  def deserialise(cls, filepath, verbose=False):
    """This is the counterpart of ``serialise``, it deserialises the model from the .nbox file.

    Args:
      nbx_path (str): path to the .nbox file
    """
    if not tarfile.is_tarfile(filepath) or not filepath.endswith(".nbox"):
      raise ValueError(f"{filepath} is not a valid .nbox file")

    with tarfile.open(filepath, "r:gz") as tar:
      folder = utils.join(gettempdir(), os.path.split(filepath)[-1].split(".")[0])
      tar.extractall(folder)

    files = glob(utils.join(folder, "*"))
    nbox_meta, nbox_meta_path, model_path = None, None, None
    for f in files:
      ext = f.split(".")[-1]
      if ext == "json":
        nbox_meta_path = f
        with open(f, "r") as f:
          nbox_meta = json.load(f)
      elif ext in ["onnx", "pkl", "pt", "torchscript"]:
        model_path = f

    logger.info(f"nbox_meta: {nbox_meta_path}")
    logger.info(f"model_path: {model_path}")
    if not (nbox_meta or model_path):
      shutil.rmtree(folder)
      raise ValueError(f"{filepath} is not a valid .nbox file")

    export_type = nbox_meta["spec"]["export_type"]
    src_framework = nbox_meta["spec"]["src_framework"]
    category = nbox_meta["spec"]["category"]

    try:
      if export_type == "onnx":
        import onnxruntime
        model = onnxruntime.InferenceSession(model_path)
      elif src_framework == "pt":
        if export_type == "torchscript":
          import torch
          model = torch.jit.load(model_path, map_location="cpu")
      elif src_framework == "sk":
        if export_type == "pkl":
          with open(model_path, "rb") as f:
            import joblib
            model = joblib.load(f)
    except Exception as e:
      raise ValueError(f"{export_type} not supported, are you missing packages? {e}")

    model = cls(model_or_model_url=model, category=category, nbox_meta=nbox_meta, verbose=verbose)
    shutil.rmtree(folder)
    return model

  def deploy(
    self,
    input_object,
    model_name=None,
    wait_for_deployment=False,
    runtime="onnx",
    deployment_type="nbox",
    deployment_id=None,
    deployment_name=None,
  ):
    """NBX-Deploy `read more <https://nimbleboxai.github.io/nbox/nbox.model.html>`_

    This deploys the current model onto our managed K8s clusters. This tight product service integration
    is very crucial for us and is the best way to make deploy a model for usage.

    Raises appropriate assertion errors for strict checking of inputs

    Args:
      input_object (Any): input to be processed
      model_name (str, optional): custom model name for this model
      cache_dir (str, optional): custom caching directory
      wait_for_deployment (bool, optional): wait for deployment to complete
      runtime (str, optional): runtime to use for deployment should be one of ``["onnx", "torchscript"]``, default is ``onnx``
      deployment_type (str, optional): deployment type should be one of ``['ovms2', 'nbox']``, default is ``nbox``
      deployment_id (str, optional): ``deployment_id`` to put this model under, if you do not pass this
        it will automatically create a new deployment check `platform <https://nimblebox.ai/oneclick>`_
        for more info or check the logs.
      deployment_name (str, optional): if ``deployment_id`` is not given and you want to create a new
        deployment group (ie. webserver will create a new ``deployment_id``) you can tell what name you
        want, be default it will create a random name.
    """
    # First Step: check the args and see if conditionals are correct or not
    def __check_conditionals():
      assert self.__framework != "nbx", "This model is already deployed on the cloud"
      assert deployment_type in ["ovms2", "nbox"], f"Only OpenVino and Nbox-Serving is supported got: {deployment_type}"
      if self.__framework == "sklearn":
        assert deployment_type == "nbox", "Only ONNX Runtime is supported for scikit-learn Framework"
      if deployment_type == "ovms2":
        assert runtime == "onnx", "Only ONNX Runtime is supported for OVMS2 Framework"
      if deployment_id and deployment_name:
        raise ValueError("deployment_id and deployment_name cannot be used together, pass only id")

    # perform sanity checks on the input values
    __check_conditionals()

    export_model_path, nbox_meta = self.serialise(
      input_object,
      model_name = model_name,
      export_type = runtime,
      generate_ov_args = deployment_type == "ovms2",
      return_meta = True
    )

    nbox_meta["spec"]["deployment_type"] = deployment_type
    nbox_meta["spec"]["deployment_id"] = deployment_id
    nbox_meta["spec"]["deployment_name"] = deployment_name

    # OCD baby!
    return deploy_model(
      export_model_path=export_model_path,
      nbox_meta=nbox_meta,
      wait_for_deployment=wait_for_deployment,
    )
