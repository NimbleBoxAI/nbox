"""
Network functions are gateway between NBX-Services.

{% CallOut variant="success" label="If you find yourself using this reach out to NimbleBox support." /%}

But for the curious mind, many of our services work on gRPC and Protobufs. This network.py
manages the quirkyness of our backend and packs multiple steps as one function.
"""

import os
import re
import grpc
import shlex
import jinja2
import fnmatch
import zipfile
import requests
from subprocess import Popen
from typing import Dict, Union
from tempfile import gettempdir
from datetime import datetime, timezone
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox import messages as mpb
from nbox.auth import secret, AuthConfig, auth_info_pb
from nbox.utils import logger, SimplerTimes
from nbox.version import __version__
from nbox.hyperloop.jobs.dag_pb2 import DAG
from nbox.init import nbox_ws_v1, nbox_grpc_stub, nbox_model_service_stub
from nbox.hyperloop.jobs.job_pb2 import  Job as JobProto
from nbox.hyperloop.common.common_pb2 import Resource, Code
from nbox.hyperloop.deploy.serve_pb2 import ModelRequest, Model
from nbox.jobs import Schedule, Serve, Job
from nbox.hyperloop.jobs.nbox_ws_pb2 import JobRequest, UpdateJobRequest
from nbox.nbxlib.operator_spec import OperatorType as OT


#######################################################################################################################
"""
# Serving

Function related to serving of any model.
"""
#######################################################################################################################


def deploy_serving(
  init_folder: str,
  serving_name: str,
  model_name: str,
  serving_id: str = None,
  workspace_id: str = None,
  resource: Resource = None,
  wait_for_deployment: bool = False,
  model_metadata: Dict[str, str] = {},
  feature_gates: Dict[str, str] = {},
  exe_jinja_kwargs: Dict[str, str] = {},
  *,
  _only_zip: bool = False,
):
  """Use the NBX-Deploy Infrastructure

  Args:
    init_folder (str): Path to the code
    serving_name (str): Name of the serving
    model_name (str): Name of the model
    serving_id (str, optional): Serving ID. Defaults to None.
    workspace_id (str, optional): Workspace ID. Defaults to None.
    resource (Resource, optional): Resource. Defaults to None.
    wait_for_deployment (bool, optional): Wait for deployment. Defaults to False.
    model_metadata (dict, optional): Model metadata. Defaults to {}.
    exe_jinja_kwargs (dict, optional): Jinja kwargs. Defaults to {}.
  """
  # check if this is a valid folder or not
  if not os.path.exists(init_folder) or not os.path.isdir(init_folder):
    raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")
  logger.info(f"Deployment ID (Name): {serving_id} ({serving_name})")

  if wait_for_deployment:
    logger.warning("Wait for deployment to be implemented, please add your own wait loop for now!")

  model_proto = Model(
    serving_group_id = serving_id,
    name = model_name,
    type = Model.ServingType.SERVING_TYPE_NBOX_OP,
    metadata = model_metadata,
    resource = resource,
    feature_gates=feature_gates
  )
  model_proto_fp = U.join(gettempdir(), "model_proto.msg")
  mpb.write_binary_to_file(model_proto, model_proto_fp)

  # zip init folder
  exe_jinja_kwargs.update({"model_name": model_name,})
  zip_path = zip_to_nbox_folder(
    init_folder = init_folder,
    id = serving_id,
    workspace_id = workspace_id,
    type = OT.SERVING,
    files_to_copy = {model_proto_fp: "model_proto.msg"},
    **exe_jinja_kwargs,
  )
  if _only_zip:
    logger.info(f"Zip file created at: {zip_path}")
    return zip_path
  else:
    return _upload_serving_zip(
      zip_path = zip_path,
      workspace_id = workspace_id,
      serving_id = serving_id,
      model_proto = model_proto,
    )


def _upload_serving_zip(zip_path: str, workspace_id: str, serving_id: str, model_proto: Model):
  model_proto.code.MergeFrom(Code(
    type = Code.Type.ZIP,
    size = int(max(os.stat(zip_path).st_size/(1024*1024), 1)) # MBs
  ))

  # get bucket URL and upload the data
  response: Model = mpb.rpc(
    nbox_model_service_stub.UploadModel,
    ModelRequest(
      model = model_proto,
      auth_info = auth_info_pb()
    ),
    "Could not get upload URL",
    raise_on_error = True
  )
  model_id = response.id
  deployment_id = response.serving_group_id
  logger.debug(f"model_id: {model_id}")
  logger.debug(f"deployment_id: {deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  s3_url = response.code.s3_url
  s3_meta = response.code.s3_meta

  use_curl = False
  if response.code.size > 10:
    logger.warning(
      f"File {zip_path} is larger than 10 MiB ({response.code.size} MiB), this might take a while\n"
      f"Switching to user/agent: cURL for this upload\n"
      f"Protip:\n"
      f"  - if you are constantly uploading large files, take a look at nbox.Relics, that's the right tool for the job"
    )
    use_curl = True

  if use_curl:
    shell_com = f'curl -X POST -F key={s3_meta["key"]} '
    for k,v in s3_meta.items():
      if k == "key":
        continue
      shell_com += f'-F {k}={v} '
    shell_com += f'-F file="@{zip_path}" {s3_url}'
    logger.debug(f"Running shell command: {shell_com}")
    out = Popen(shlex.split(shell_com)).wait()
    if out != 0:
      logger.error(f"Failed to upload model: {out}, please check logs for this")
      return
  else:
    r = requests.post(url=s3_url, data=s3_meta, files={"file": (s3_meta["key"], open(zip_path, "rb"))})
    try:
      r.raise_for_status()
    except:
      logger.error(f"Failed to upload model: {r.content.decode('utf-8')}")
      return

  # model is uploaded successfully, now we need to deploy it
  logger.info(f"Model uploaded successfully, deploying ...")
  response: Model = mpb.rpc(
    nbox_model_service_stub.Deploy,
    ModelRequest(
      model = Model(
        id = model_id,
        serving_group_id = deployment_id,
        resource = model_proto.resource,
        feature_gates = model_proto.feature_gates,
      ),
      auth_info = auth_info_pb(),
    ),
    "Could not deploy model",
    raise_on_error=True
  )

  # write out all the commands for this deployment
  # logger.info("API will soon be hosted, here's how you can use it:")
  # _api = f"Operator.from_serving('{serving_id}', $NBX_TOKEN, '{workspace_id}')"
  # _cli = f"python3 -m nbox serve forward --id_or_name '{serving_id}' --workspace_id '{workspace_id}'"
  # _curl = f"curl https://api.nimblebox.ai/{serving_id}/forward"
  _webpage = f"{secret(AuthConfig.url)}/workspace/{workspace_id}/deploy/{serving_id}"
  # logger.info(f" [python] - {_api}")
  # logger.info(f"    [CLI] - {_cli} --token $NBX_TOKEN --args")
  # logger.info(f"   [curl] - {_curl} -H 'NBX-KEY: $NBX_TOKEN' -H 'Content-Type: application/json' -d " + "'{}'")
  logger.info(f"  [page] - {_webpage}")

  return Serve(serving_id = serving_id, model_id = model_id)


#######################################################################################################################
"""
# Jobs

Function related to batch processing of any model.
"""
#######################################################################################################################


def deploy_job(
  init_folder: str,
  job_name: str,
  dag: DAG,
  schedule: Schedule,
  resource: Resource,
  workspace_id: str = None,
  job_id: str = None,
  feature_gates: Dict[str, str] = None,
  exe_jinja_kwargs = {},
  *,
  _only_zip: bool = False,
  _unittest: bool = False
) -> None:
  """Upload code for a NBX-Job.

  Args:
    init_folder (str, optional): Name the folder to zip
    job_id_or_name (Union[str, int], optional): Name or ID of the job
    dag (DAG): DAG to upload
    workspace_id (str): Workspace ID to deploy to, if not specified, will use the personal workspace
    schedule (Schedule, optional): If ``None`` will run only once, else will schedule the job
    cache_dir (str, optional): Folder where to put the zipped file, if ``None`` will be ``tempdir``
  Returns:
    Job: Job object
  """
  # check if this is a valid folder or not
  if not os.path.exists(init_folder) or not os.path.isdir(init_folder):
    raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")
  if (job_name is None or job_name == "") and job_id == "":
    raise ValueError("Please specify a job name or ID")

  # logger.debug(f"deploy_job:\n  init_folder: {init_folder}\n  name: {job_name}\n  id: {job_id}")

  # job_id, job_name = _get_job_data(name = job_name, id = job_id, workspace_id = workspace_id)
  logger.info(f"Job name: {job_name}")
  logger.info(f"Job ID: {job_id}")

  # intialise the console logger
  URL = secret("nbx_url")
  logger.debug(f"Schedule: {schedule}")
  logger.debug("-" * 30 + " NBX Jobs " + "-" * 30)
  logger.debug(f"Deploying on URL: {URL}")

  # create the proto for this Operator
  job_proto = JobProto(
    id = job_id,
    name = job_name or U.get_random_name(True).split("-")[0],
    created_at = SimplerTimes.get_now_pb(),
    schedule = schedule.get_message() if schedule is not None else None, # JobProto.Schedule(cron = "0 0 24 2 0"),
    dag = dag,
    resource = resource,
    feature_gates=feature_gates
  )
  job_proto_fp = U.join(gettempdir(), "job_proto.msg")
  mpb.write_binary_to_file(job_proto, job_proto_fp)

  if _unittest:
    return job_proto

  # zip the entire init folder to zip
  zip_path = zip_to_nbox_folder(
    init_folder = init_folder,
    id = job_id,
    workspace_id = workspace_id,
    type = OT.JOB,
    files_to_copy = {job_proto_fp: "job_proto.msg"},
    **exe_jinja_kwargs,
  )
  return _upload_job_zip(zip_path, job_proto,workspace_id)

def _upload_job_zip(zip_path: str, job_proto: JobProto, workspace_id: str):
  # determine if it's a new Job based on GetJob API
  try:
    old_job_proto: JobProto = nbox_grpc_stub.GetJob(
      JobRequest(job = JobProto(id = job_proto.id), auth_info = auth_info_pb())
    )
    new_job = old_job_proto.status in [JobProto.Status.NOT_SET, JobProto.Status.ARCHIVED]
  except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.NOT_FOUND:
      new_job = True
      old_job_proto = JobProto()
    else:
      raise e

  if not new_job:
    # incase an old job exists, we need to update few things with the new information
    logger.debug("Found existing job, checking for update masks")
    paths = []
    mask = mpb.field_mask(old_job_proto, job_proto)
    if old_job_proto.resource.SerializeToString(deterministic = True) != job_proto.resource.SerializeToString(deterministic = True):
      paths.append("resource")
    if old_job_proto.schedule.cron != job_proto.schedule.cron:
      paths.append("schedule.cron")
    logger.debug(f"Updating fields: {paths}")
    nbox_grpc_stub.UpdateJob(
      UpdateJobRequest(job = job_proto, update_mask = FieldMask(paths=paths), auth_info=auth_info_pb()),
    )

  # update the JobProto with file sizes
  job_proto.code.MergeFrom(Code(
    size = max(int(os.stat(zip_path).st_size / (1024 ** 2)), 1), # jobs in MB
    type = Code.Type.ZIP,
  ))

  # UploadJobCode is responsible for uploading the code of the job
  response: JobProto = mpb.rpc(
    nbox_grpc_stub.UploadJobCode,
    JobRequest(job = job_proto, auth_info=auth_info_pb()),
    f"Failed to upload job: {job_proto.id} | {job_proto.name}"
  )
  job_proto.MergeFrom(response)
  s3_url = job_proto.code.s3_url
  s3_meta = job_proto.code.s3_meta
  
  logger.info(f"Uploading model to S3 ... (fs: {response.code.size:0.3f} MB)")
  use_curl = False
  if response.code.size > 10:
    logger.info(
      f"File {zip_path} is larger than 10 MB ({response.code.size} MB), this might take a while\n"
      f"Switching to user/agent: cURL for this upload\n"
      f"Protip:\n"
      f"  - if you are constantly uploading large files, take a look at nbox.Relics, that's the right tool for the job"
    )
    use_curl = True

  if use_curl:
    shell_com = f'curl -X POST -F key={s3_meta["key"]} '
    for k,v in s3_meta.items():
      if k == "key":
        continue
      shell_com += f'-F {k}={v} '
    shell_com += f'-F file="@{zip_path}" {s3_url}'
    logger.debug(f"Running shell command: {shell_com}")
    out = Popen(shlex.split(shell_com)).wait()
    if out != 0:
      logger.error(f"Failed to upload model: {out}, please check logs for this")
      return
  else:
    r = requests.post(url=s3_url, data=s3_meta, files={"file": (s3_meta["key"], open(zip_path, "rb"))})
    try:
      r.raise_for_status()
    except:
      logger.error(f"Failed to upload model: {r.content.decode('utf-8')}")
      return

  job_proto.feature_gates.update({
    "UsePipCaching": "", # some string does not honour value
    "EnableAuthRefresh": ""
  })
  auth_info = auth_info_pb()
  if new_job:
    logger.info("Creating a new job")
    mpb.rpc(
      stub = nbox_grpc_stub.CreateJob,
      message = JobRequest(job = job_proto, auth_info = auth_info),
      err_msg = "Failed to create job"
    )

  if not old_job_proto.feature_gates:
    logger.info("Updating feature gates")
    mpb.rpc(
      stub = nbox_grpc_stub.UpdateJob,
      message = UpdateJobRequest(job = job_proto, update_mask = FieldMask(paths = ["feature_gates"]), auth_info = auth_info),
      err_msg = "Failed to update job",
      raise_on_error = False
    )

  # write out all the commands for this job
  # logger.info("Run is now created, to 'trigger' programatically, use the following commands:")
  # _api = f"nbox.Job(id = '{job_proto.id}', workspace_id='{job_proto.auth_info.workspace_id}').trigger()"
  # _cli = f"python3 -m nbox jobs --job_id {job_proto.id} --workspace_id {workspace_id} trigger"
  # _curl = f"curl -X POST {secret(AuthConfig.url)}/api/v1/workspace/{job_proto.auth_info.workspace_id}/job/{job_proto.id}/trigger"
  # logger.info(f" [python] - {_api}")
  # logger.info(f"    [CLI] - {_cli}")
  # logger.info(f"   [curl] - {_curl} -H 'authorization: Bearer $NBX_TOKEN' -H 'Content-Type: application/json' -d " + "'{}'")

  _webpage = f"{secret(AuthConfig.url)}/workspace/{workspace_id}/jobs/{job_proto.id}"
  logger.info(f"   [page] - {_webpage}")

  # create a Job object and return so CLI can do interesting things
  return Job(job_id = job_proto.id)


#######################################################################################################################
"""
# Common

Function related to both NBX-Serving and NBX-Jobs
"""
#######################################################################################################################

def zip_to_nbox_folder(
  init_folder: str,
  id: str,
  workspace_id: str,
  type: Union[OT.JOB, OT.SERVING],
  files_to_copy: Dict[str, str] = {},
  **jinja_kwargs
):
  """
  This function creates the zip file that is ulitimately uploaded to NBX-Serving or NBX-Jobs.

  Args:
    init_folder (str): The folder that contains the files to be zipped
    id (str): The id of the job or serving
    workspace_id (str): The workspace id of the job or serving
    type (OT_TYPE): The type of the job or serving, comes from nbxlib.operator_spec.OT_TYPE
    files_to_copy (Dict[str, str], optional): A dictionary of source to target filepaths to copy to the zip file. Defaults to {}.
    **jinja_kwargs: Any additional arguments to be passed to the jinja template
  """
  # zip all the files folder
  all_f = U.get_files_in_folder(init_folder, followlinks = False)

  # find a .nboxignore file and ignore items in it
  to_ignore_pat = []
  to_ignore_folder = []
  for f in all_f:
    if f.split("/")[-1] == ".nboxignore":
      with open(f, "r") as _f:
        for pat in _f:
          pat = pat.strip()
          if pat.endswith("/"):
            to_ignore_folder.append(pat)
          else:
            to_ignore_pat.append(pat)
      break

  # print(all_f)

  # print("to_ignore_pat:", to_ignore_pat)
  # print("to_ignore_folder:", to_ignore_folder)

  # two different lists for convinience
  to_remove = []
  for ignore in to_ignore_pat:
    if "*" in ignore:
      x = fnmatch.filter(all_f, ignore)
    else:
      x = [f for f in all_f if f.endswith(ignore)]
    to_remove.extend(x)
  to_remove_folder = []
  for ignore in to_ignore_folder:
    for f in all_f:
      if re.search(ignore, f):
        to_remove_folder.append(f)
  to_remove += to_remove_folder
  all_f = [x for x in all_f if x not in to_remove]
  logger.info(f"Will zip {len(all_f)} files")
  # print(all_f)
  # exit()
  # print(to_remove)
  # exit()

  # zip all the files folder
  zip_path = U.join(gettempdir(), f"nbxjd_{id}@{workspace_id}.nbox")
  logger.info(f"Packing project to '{zip_path}'")
  with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    abspath_init_folder = os.path.abspath(init_folder)
    for f in all_f:
      arcname = f[len(abspath_init_folder)+1:]
      logger.debug(f"Zipping {f} => {arcname}")
      zip_file.write(f, arcname = arcname)

    # create the exe.py file
    exe_jinja_path = U.join(U.folder(__file__), "assets", "exe.jinja")
    exe_path = U.join(gettempdir(), "exe.py")
    logger.debug(f"Writing exe to: {exe_path}")
    with open(exe_jinja_path, "r") as f, open(exe_path, "w") as f2:
      # get a timestamp like this: Monday W34 [UTC 12 April, 2022 - 12:00:00]
      _ct = datetime.now(timezone.utc)
      _day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][_ct.weekday()]
      created_time = f"{_day} W{_ct.isocalendar()[1]} [ UTC {_ct.strftime('%d %b, %Y - %H:%M:%S')} ]"

      # fill up the jinja template
      code = jinja2.Template(f.read()).render({
        "created_time": created_time,
        "nbox_version": __version__,
        **jinja_kwargs
      })
      f2.write(code)
    # print(os.stat(exe_path))

    zip_file.write(exe_path, arcname = "exe.py")
    for src, trg in files_to_copy.items():
      zip_file.write(src, arcname = trg)

  return zip_path
