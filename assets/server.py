# this is the deprecated server since >v0.8.5

import os
import sys
import time
import json
import requests
import subprocess
import fastapi as fa
from typing import Dict, Any
from pydantic import BaseModel
from functools import lru_cache
from tempfile import gettempdir
from starlette.requests import Request
from starlette.responses import Response

# extract the tar file
fpath = os.getenv("NBOX_MODEL_PATH", None)
if fpath == None:
  raise ValueError("have you set env var: NBOX_MODEL_PATH")

import tarfile
if not tarfile.is_tarfile(fpath) or not fpath.endswith(".nbox"):
  raise ValueError(f"{fpath} is not a valid .nbox file")

with tarfile.open(fpath, "r:gz") as tar:
  folder = os.path.join(gettempdir(), os.path.basename(fpath).replace(".nbox", ""))
  tar.extractall(folder)

if os.environ.get("NBOX_INSTALL_DEP", False):
  subprocess.run(["pip", "install", "-r", os.path.join(folder, "requirements.txt")])

with open(f"{folder}/nbox_config.json", "r") as f:
  config = json.load(f)

# save the ~/.nbx/secrets.json
r = requests.get("https://raw.githubusercontent.com/NimbleBoxAI/nbox/master/assets/sample_config.json")
r.raise_for_status()
home_dir = os.path.join(os.path.expanduser("~"), ".nbx")
os.makedirs(home_dir, exist_ok=True)
with open(os.path.join(home_dir, "secrets.json"), "wb") as f:
  sys.stdout.write(r.content.decode())
  f.write(r.content)

# this is where nbox gets loaded
os.environ["NBOX_HOME_DIR"] = home_dir # no need but still
os.environ["NBOX_JSON_LOG"] = "1"
from nbox.model import Model
from nbox.utils import logger
SERVING_MODE = os.path.splitext(fpath)[1][1:]
model: Model = Model.deserialise(folder=folder, model_spec=config)
if hasattr(model.model, "eval"):
  model.model.eval()

from nbox.messages import message_to_dict


class ModelInput(BaseModel):
  inputs: Any
  method: str = None
  input_dtype: str = None
  message: str = None

class ModelOutput(BaseModel):
  outputs: Any
  time: int
  message: str = None

class MetadataModel(BaseModel):
  time: int
  metadata: Dict[str, Any]

class PingRespose(BaseModel):
  time: int
  message: str = None

@lru_cache(1) # fetch only once
def nbox_meta():
  data = message_to_dict(model.model_spec,)
  return data

app = fa.FastAPI()

# add route for /
@app.get("/", status_code=200, response_model=PingRespose)
async def ping(r: Request, response: Response):
  return dict(time=int(time.time()), message="pong")

# add route for /metadata
@app.get("/metadata", status_code=200, response_model=MetadataModel)
async def get_meta(r: Request, response: Response):
  return dict(time=int(time.time()), metadata = nbox_meta())

# add route for /predict
@app.post("/predict", status_code=200, response_model=ModelOutput)
async def predict(r: Request, response: Response, item: ModelInput):
  logger.debug(str(item.inputs)[:100])

  try:
    output = model(item.inputs)
  except Exception as e:
    response.status_code = 500
    logger.error(f"error: {str(e)}")
    return {"message": str(e), "time": int(time.time())}

  try:
    json.dumps(output)
  except Exception as e:
    response.status_code = 400
    logger.error("user_error: output is not JSON serializable")
    return {
      "message": "Output is not JSON serializable! Please redeploy with proper post_fn.",
      "time": int(time.time())
    }

  return {
    "outputs": output,
    "time": int(time.time())
  }
