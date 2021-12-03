import os
import time
import fastapi as fa
from typing import Dict, Any
from pydantic import BaseModel
from starlette.responses import Response
from starlette.requests import Request

from nbox.model import Model
from nbox.utils import VERSION, convert_to_list

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
    modelSpec: Dict[str, Any]
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
        modelSpec={
            "model_path": fpath,
            "source_serving": SERVING_MODE,
            "nbox_version": VERSION,
        },
        metadata = {
            "signature_def": {
                "signatureDef": {
                    "serving_default": model.nbox_meta,
                }
            }
        }
    )

@app.post("/predict", status_code=200, response_model=ModelOutput)
def predict(r: Request, response: Response, item: ModelInput):
    # look at the inputs
    print(">=>=>=> nbox serving request")
    print("Got input ::", str(item.inputs)[:100])

    try:
        output = model(item.inputs, method = item.method, return_dict = True)
    except Exception as e:
        response.status_code = 500
        return {"message": str(e), "time": int(time.time())}

    if isinstance(output, str):
        response.status_code = 400
        print(">=>=>=> nbox serving closed")
        return {"message": output, "time": int(time.time())}

    try:
        output = convert_to_list(output)
    except Exception as e:
        print("convert_to_list failed")
        response.status_code = 500
        return {"message": str(e), "time": int(time.time())}

    print(">=>=>=> nbox serving closed")
    response.status_code = 200
    return {"outputs": output, "time": int(time.time())}
