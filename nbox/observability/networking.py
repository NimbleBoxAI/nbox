"""This file contains the code for observability in the networking layer."""

from typing import Tuple
import time

try:
  import starlette
  from starlette.requests import Request
  from starlette.responses import Response
  from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
  from starlette.routing import Match
  from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
  from starlette.types import ASGIApp
except ImportError:
  starlette = None


class GenericMetric:
  # this is like the prometheus client's Counter, Gauge, Histogram, etc. but instead of client
  # worrying about the types, here we are saying that it's the servers headache to worry about
  def __init__(self, name: str):
    pass


class AsgiLmaoMiddleware(BaseHTTPMiddleware):
  def __init__(self, app: ASGIApp, filter_unhandled_paths: bool = False) -> None:
    if starlette is None:
      raise ValueError("Starlette is not installed. pip install -U nbox\[serving\]")
    super().__init__(app)
    self.filter_unhandled_paths = filter_unhandled_paths

  async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
    method = request.method
    path_template, is_handled_path = self.get_path_template(request)

    if self._is_path_filtered(is_handled_path):
      return await call_next(request)

    REQUESTS_IN_PROGRESS.labels(method = method, path_template = path_template).inc()
    REQUESTS.labels(method=method, path_template=path_template).inc()
    before_time = time.perf_counter()
    try:
      response = await call_next(request)
    except BaseException as e:
      status_code = HTTP_500_INTERNAL_SERVER_ERROR
      EXCEPTIONS.labels(method=method, path_template=path_template, exception_type=type(e).__name__).inc()
      raise e from None
    else:
      status_code = response.status_code
      after_time = time.perf_counter()
      REQUESTS_PROCESSING_TIME.labels(method=method, path_template=path_template).observe(after_time - before_time)
    finally:
      RESPONSES.labels(method=method, path_template=path_template, status_code=status_code).inc()
      REQUESTS_IN_PROGRESS.labels(method=method, path_template=path_template).dec()

    return response

  @staticmethod
  def get_path_template(request: Request) -> Tuple[str, bool]:
    for route in request.app.routes:
      match, child_scope = route.matches(request.scope)
      if match == Match.FULL:
        return route.path, True
    return request.url.path, False

  def _is_path_filtered(self, is_handled_path: bool) -> bool:
    return self.filter_unhandled_paths and not is_handled_path





def metrics(request: Request) -> Response:
  if "prometheus_multiproc_dir" in os.environ:
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
  else:
    registry = REGISTRY

  return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})



from starlette_prometheus import metrics, PrometheusMiddleware
