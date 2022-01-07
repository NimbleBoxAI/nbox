# logger/

import json
import logging

from pythonjsonlogger.jsonlogger import JsonFormatter, RESERVED_ATTRS

class CustomJsonFormatter(JsonFormatter):
      def format(self, record):
        data = record.__dict__.copy()
        mydict = {
          "levelname": data.pop("levelname"),
          "created": data.pop("created"),
          "loc": f'{data.pop("filename")}:{data.pop("lineno")}',
          "message": data.pop("msg"),
        }
        for reserved in RESERVED_ATTRS:
            if reserved in data:
                del data[reserved]
        mydict.update(data)
        return json.dumps(mydict)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logHandler = logging.StreamHandler()
logHandler.setFormatter(CustomJsonFormatter())
logger.addHandler(logHandler)
del logger.handlers[0]

print("333333", logger.handlers)

# /logger

import os
import time
import fastapi as fa
from typing import Dict, Any
from pydantic import BaseModel
from starlette.responses import Response
from starlette.requests import Request

import nbox
from nbox.model import Model
from nbox.utils import convert_to_list

fpath = os.getenv("NBOX_MODEL_PATH", None)
if fpath == None:
    raise ValueError("have you set env var: NBOX_MODEL_PATH")

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
    spec: Dict[str, Any]
    metadata: Dict[str, Any]

class PingRespose(BaseModel):
    time: int
    message: str = None


app = fa.FastAPI()
SERVING_MODE = os.path.splitext(fpath)[1][1:]
model = Model.deserialise(fpath, verbose = True)

# add route for /
@app.get("/", status_code=200, response_model=PingRespose)
def ping(r: Request, response: Response):
    return {"time": int(time.time()), "message": "pong"}

# add route for /metadata
@app.get("/metadata", status_code=200, response_model=MetadataModel)
def get_meta(r: Request, response: Response):
    return dict(
        time=int(time.time()),
        spec={
            "model_path": fpath,
            "source_serving": SERVING_MODE,
            "nbox_version": nbox.__version__,
            "name": model.nbox_meta["spec"]["model_name"],
            **model.nbox_meta["spec"]
        },
        metadata = model.nbox_meta["metadata"]
    )

@app.post("/predict", status_code=200, response_model=ModelOutput)
def predict(r: Request, response: Response, item: ModelInput):
    logger.info(str(item.inputs)[:100], extra = {"_time": int(time.time())})

    try:
        output = model(item.inputs, method = item.method, return_dict = True)
    except Exception as e:
        response.status_code = 500
        logger.error('{"error_0": {}, "name": {}}'.format(str(e), "model_predict"))
        return {"message": str(e), "time": int(time.time())}

    if isinstance(output, str):
        response.status_code = 400
        logger.error('{"error_1": {}, "name": {}}'.format(output, "incorrect_predict"))
        return {"message": output, "time": int(time.time())}

    try:
        output = convert_to_list(output)
    except Exception as e:
        response.status_code = 500
        logger.error('{"error_2": {}, "name": {}}'.format(str(e), "serialise"))
        return {"message": str(e), "time": int(time.time())}

    response.status_code = 200
    return {"outputs": output, "time": int(time.time())}
