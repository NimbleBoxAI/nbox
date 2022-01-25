import json
from time import time
import requests
from logging import getLogger
logger = getLogger()

from .common import IllegalFormatError, ModelMeta
from ..auth import secret

def load_model(model, nbx_api_key):
  def load_condition():
    if not model.startswith("https://") or model.startswith("http://"):
      raise IllegalFormatError("Model URL must start with http:// or https://")
    if isinstance(nbx_api_key, str):
      raise IllegalFormatError("Nbx API key must be a string")
    if nbx_api_key.startswith("nbxdeploy_"):
      raise IllegalFormatError("Not a valid NBX Api key, please check again.")
    if model.startswith("http"):
      raise IllegalFormatError("Are you sure this is a valid URL?")

  load_condition()
  logger.info(f"Trying to load from url: {model}")

  # fetch the metadata from the cloud
  model_url = model.rstrip("/")
  logger.info("Getting model metadata")
  URL = secret.get("nbx_url")
  r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={model}&key={nbx_api_key}")
  try:
    r.raise_for_status()
  except:
    raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

  # start getting the metadata, note that we have completely dropped using OVMS meta and instead use nbox_meta
  content = json.loads(r.content.decode())["meta"]
  nbox_meta = content["nbox_meta"]

  all_inputs = nbox_meta["metadata"]["inputs"]
  templates = {}
  for node, meta in all_inputs.items():
    templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]
  logger.info("Cloud infer metadata obtained")

  category = nbox_meta["spec"]["category"]

  # if category is "text" or if it is dict then any key is "text"
  tokenizer = None
  max_len = None
  if category == "text" or (isinstance(category, dict) and any([x == "text" for x in category.values()])):
    import transformers

    model_key = nbox_meta["spec"]["model_key"].split("::")[0].split("transformers/")[-1]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
    max_len = templates["input_ids"][-1]

  image_parser = ImageParser(cloud_infer=True, post_proc_fn=lambda x: x.tolist(), templates=templates)
  text_parser = TextParser(tokenizer=tokenizer, max_len=max_len, post_proc_fn=lambda x: x.tolist())

  def forward_pass(meta: ModelMeta):
    logger.info(f"Hitting API: {self.model_or_model_url}")
    st = time()
    # OVMS has :predict endpoint and nbox has /predict
    _p = "/" if "export_type" in self.nbox_meta["spec"] else ":"
    json = {"inputs": model_input}
    if "export_type" in self.nbox_meta["spec"]:
      json["method"] = method
    r = requests.post(model_url + f"/{_p}predict", json=json, headers={"NBX-KEY": nbx_api_key})
    et = time() - st
    out = None

    try:
      r.raise_for_status()
      out = r.json()

      # first try outputs is a key and we can just get the structure from the list
      if isinstance(out["outputs"], dict):
        out = {k: np.array(v) for k, v in r.json()["outputs"].items()}
      elif isinstance(out["outputs"], list):
        out = np.array(out["outputs"])
      else:
        raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")
      logger.info(f"Took {et:.3f} seconds!")
    except Exception as e:
      logger.info(f"Failed: {str(e)} | {r.content.decode()}")

  return forward_pass

class NBXDeployMixin:
  load_model = load_model
