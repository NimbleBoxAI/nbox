"""
Jobs
====

``nbox.Job`` is a wrapper to the APIs that's it.
"""

import sys
import jinja2

from ..utils import logger
from .. import utils as U
from ..init import nbox_grpc_stub
from ..network import Cron
from ..auth import secret

################################################################################
# NBX-Jobs Functions
# ==================
# These functions are assigned as static functions to the ``nbox.Job`` class.
################################################################################

def new(project_name):
  import os, re
  from datetime import datetime
  created_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

  out = re.findall("^[a-zA-Z_]+$", project_name)
  if not out:
    raise ValueError("Project name can only contain letters and underscore")

  if os.path.exists(project_name):
    raise ValueError(f"Project {project_name} already exists")

  # ask user requirements here for customisation
  run_on_build = input("> Is this a run on build job? (y/N) ").lower() == "y"

  scheduled = None
  instance = None
  if run_on_build:
    # in this case the job will be run on a nbx-build instance (internal testing)
    # real advantage is ability to run on GPUs and persistent storage
    logger.info("This job will run on NBX-Build")
    instance = input("> Instance name or ID: ").strip()
  else:
    logger.info("This job will run on NBX-Jobs")
    scheduled = input("> Is this a recurring job (y/N)? ").lower() == "y"
    if scheduled:
      logger.info("This job will be scheduled to run on a recurring basis")
    else:
      logger.info(f"This job will run only once")
  
  logger.info(f"Creating a folder: {project_name}")
  os.mkdir(project_name)
  os.chdir(project_name)

  # jinja is cool
  assets = U.join(U.folder(U.folder(__file__)), "assets")
  path = U.join(assets, "job_new.jinja")
  with open(path, "r") as f, open("exe.py", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        run_on_build = run_on_build,
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
        instance = instance
    ))

  path = U.join(assets, "job_new_readme.jinja")
  with open(path, "r") as f, open("README.md", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
    ))

  open("requirements.txt", "w").close() # ~ touch requirements.txt

  logger.debug("Completed")

def open():
  import webbrowser
  webbrowser.open(secret.get("nbx_url")+"/"+"jobs")

def get_job_list(workspace_id: str):
  import grpc
  from ..hyperloop.job_pb2 import NBXAuthInfo
  from ..hyperloop.nbox_ws_pb2 import ListJobsRequest
  from google.protobuf.json_format import MessageToDict
  
  auth_info = NBXAuthInfo(workspace_id = workspace_id)
  
  try:
    out = nbox_grpc_stub.ListJobs(ListJobsRequest(auth_info = auth_info))
  except grpc.RpcError as e:
    logger.error(f"{e.details()}")
    sys.exit(1)

  out = MessageToDict(out)
  print(out)


################################################################################
# NimbleBox.ai Jobs
# =================
# This is the actual job object that users can manipulate. It is a shallow class
# around the NBX-Jobs gRPC API.
################################################################################

class Job:
  def __init__(self, id, workspace_id =None):
    from ..hyperloop.job_pb2 import NBXAuthInfo

    self.id = id
    self.workspace_id = workspace_id
    self.auth_info = NBXAuthInfo(workspace_id = self.workspace_id)
    self.update()

  # static methods
  new = staticmethod(new)
  open = staticmethod(open)
  status = staticmethod(get_job_list)

  def change_schedule(self, new_schedule: Cron):
    # nbox should only request and server should check if possible or not
    pass

  def stream_logs(self, f = sys.stdout):
    # this function will stream the logs of the job in anything that can be written to
    import grpc
    from ..hyperloop.nbox_ws_pb2 import JobLogsRequest
    
    logger.info(f"Streaming logs of job {self.id}")
    try:
      log_iter = nbox_grpc_stub.GetJobLogs(JobLogsRequest(job = self._this_job))
    except grpc.RpcError as e:
      logger.error(f"Could not get logs of job {self.id}")
      raise e

    for job_log in log_iter:
      for log in job_log.log:
        f.write(log)
        f.flush()

  def delete(self):
    import grpc
    from ..hyperloop.nbox_ws_pb2 import JobInfo
    try:
      nbox_grpc_stub.DeleteJob(JobInfo(job = self._this_job,))
    except grpc.RpcError as e:
      logger.error(f"Could not delete job {self.id}")
      raise e

  def update(self):
    import grpc
    from ..hyperloop.nbox_ws_pb2 import JobInfo
    from ..hyperloop.job_pb2 import Job as JobProto
    logger.info("Updating job info")

    try:
      job: JobProto = nbox_grpc_stub.GetJob(JobInfo(job = JobProto(id = self.id, auth_info = self.auth_info)))
    except grpc.RpcError as e:
      logger.error(f"Could not get job {id}")
      raise e
    for descriptor, value in job.ListFields():
      setattr(self, descriptor.name, value)
    self._this_job = job
    self._this_job.auth_info.CopyFrom(self.auth_info)

  def trigger(self):
    import grpc
    from nbox.hyperloop.nbox_ws_pb2 import JobInfo
    logger.info(f"Triggering job {self.id}")
    
    try:
      
      nbox_grpc_stub.TriggerJob(JobInfo(job=self._this_job))
    except grpc.RpcError as e:
      logger.error(f"Could not trigger job {self.id}")
      raise e

