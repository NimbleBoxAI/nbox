import json
from time import time
import requests
from logging import getLogger
logger = getLogger()

from .common import *
from ..auth import secret

class NBXModel(FrameworkAgnosticModel):
  def __init__(self, url, key):
    self.url = url
    self.key = key

    logger.info(f"Trying to load as url")
    def load_condition():
      if not isinstance(url, str):
        raise IllegalFormatError(f"Model must be a string, got: {type(url)}")
      if not (url.startswith("https://") or url.startswith("http://")):
        raise IllegalFormatError("Model URL must start with http:// or https://")
      if not isinstance(key, str):
        raise IllegalFormatError("Nbx API key must be a string")
      if not key.startswith("nbxdeploy_"):
        raise IllegalFormatError("Not a valid NBX Api key, please check again.")

    load_condition()

    # fetch the metadata from the cloud
    model_url = url.rstrip("/")
    logger.info("Getting model metadata")
    URL = secret.get("nbx_url")
    r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={model_url}&key={key}")
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

  def forward(self, model_input):
    logger.info(f"Hitting API: {self.model_or_model_url}")
    st = time()
    # OVMS has :predict endpoint and nbox has /predict
    _p = "/" if "export_type" in self.nbox_meta["spec"] else ":"
    json = {"inputs": model_input}
    if "export_type" in self.nbox_meta["spec"]:
      json["method"] = method
    r = requests.post(self.url + f"/{_p}predict", json=json, headers={"NBX-KEY": self.key = key})
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

  def serialise(self):
    raise IllegalFormatError("Cannot export an already deployed model")
