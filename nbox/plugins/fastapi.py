from uuid import uuid4
import json

from nbox import logger, lo, Project
from nbox.utils import SimplerTimes
from nbox.plugins.base import import_error

try:
  from fastapi import FastAPI, Response, Request
  from fastapi.middleware import Middleware
except ImportError:
  raise import_error("fastapi")

# so we need a mixture of both the middleware and the decorator because the decorator can modify the response
# but the middleware can do things after response is sent

async def _decorator_middleware(request: Request, call_next):
  response = await call_next(request)
  scope = request.scope
  trace_id = scope.get("nbx_trace_id", "")
  body = b'{"id":"'+ trace_id.encode("utf-8") + b'","out":'
  async for chunk in response.body_iterator:
    body += chunk
  body += b"}"
  body = body.decode("utf-8")
  response.headers["content-length"] = str(len(body))
  response.headers["content-type"] = "application/json"
  return Response(
    content=body,
    status_code=response.status_code,
    headers=response.headers,
    media_type=response.media_type,
    background=response.background,
  )

class LmaoAsgiMiddleware(Middleware):
  def __init__(self, app, tracker):
    self.app = app
    self.tracker = tracker

  # TODO: @yashbonde add a buffer uploading mechanism for the logs once gRPC is ready

  async def __call__(self, scope, receive, send):
    _to_log = scope['type'] == 'http' and scope["path"] not in ["/", "/metadata"]
    if _to_log:
      trace_id = str(uuid4())
      scope["nbx_trace_id"] = trace_id
      st = SimplerTimes.get_now_ns()
    await self.app(scope, receive, send)
    if _to_log:
      data = {
        "id": trace_id,
        "path": scope["path"],
        "latency": SimplerTimes.get_now_ns() - st
      }
      logger.debug(lo("api_log:", data = data))
      if self.tracker:
        self.tracker.log({
          "data": json.dumps(data),
        })


def add_live_tracker(project: Project, app:FastAPI, metadata = {}) -> FastAPI:
  live_tracker = project.get_live_tracker(metadata = metadata)
  app.add_middleware(LmaoAsgiMiddleware, tracker = live_tracker)
  app.middleware("http")(_decorator_middleware)
  return app
