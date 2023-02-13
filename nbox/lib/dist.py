"""
With NimbleBox you can run cluster wide workloads from anywhere. This requires capabilities around distributed computing,
process managements. The code here is tested along with `nbox.Relic` to perform distributed local and cloud processing.

{% CallOut variant="success" label="If you find yourself using this reach out to NimbleBox support." /%}
"""

import os
import json
from pprint import pformat

import nbox.utils as U
from nbox.utils import logger
from nbox.relics import Relics
from nbox.operator import Operator
from nbox.auth import secret, ConfigString
from nbox.nbxlib.tracer import Tracer
from nbox.hyperloop.jobs.job_pb2 import Job
from nbox.nbxlib.serving import serve_operator, SupportedServingTypes as SST, ServingMetadata
from nbox.messages import read_file_to_binary
from nbox.hyperloop.deploy.serve_pb2 import Model as ModelProto

from nbox.lmao import LMAO_JOB_TYPE_PREFIX, _lmaoConfig, get_lmao_stub, ExperimentConfig
from nbox.sublime.proto.lmao_pb2 import Run, ListProjectsRequest, ListProjectsResponse

# Manager
class LocalNBXLet(Operator):
  def __init__(self, op: Operator, in_key: str, out_key: str):
    super().__init__()
    self.op = op
    self.in_key = in_key
    self.out_key = out_key

  def __repr__(self):
    return f"LocalNBXLet({self.op.__qualname__}, {self.in_key}, {self.out_key})"

  def forward(self):
    x = U.from_pickle(self.in_key)
    y = self.op(*x)
    U.to_pickle(y, self.out_key)


class NBXLet(Operator):
  def __init__(self, op: Operator):
    """The Operator that runs the things on any pod on the NimbleBox Jobs + Deploy platform.
    Name a parody of kubelet, dockerlet, raylet, etc"""
    super().__init__()
    self.op = op

  def run(self):
    """Run this as a batch process"""
    status = Job.Status.ERROR
    try:
      tracer = Tracer()
      if hasattr(self.op._tracer, "job_proto"):
        self.op.thaw(self.op._tracer.job_proto)
      workspace_id = tracer.workspace_id
      logger.info(f"Workspace Id: {workspace_id}")

      # this is important since nbox uses ConfigString.workspace_id place to get workspace_id from while the init_container
      # might place it at a different place. as of this writing, init_container -> "workspace_id" and nbox -> "config.global.workspace_id"
      secret.put(ConfigString.workspace_id, workspace_id, True) 

      job_id = tracer.job_id
      self.op.propagate(_tracer = tracer)

      # get the user defined tag 
      run_tag = os.getenv("NBOX_RUN_METADATA", "")
      logger.info(f"Tag: {run_tag}")

      # in the NimbleBox system we provide tags for each key which essentially tells what is the behaviour
      # of the job. For example if it contains the string LMAO which means we need to initialise a couple
      # of things, or this can be any other job type
      if run_tag.startswith(LMAO_JOB_TYPE_PREFIX):
        # originally we had a strategy to use Relics to store the information about the initialisation and passed args
        # however we are not removing that because we don't want to spend access money when we are anyways storing all
        # the information in the LMAO DB. so now we get the details of the run and get all the information from there.
        _lmao_stub = get_lmao_stub()
        project_id, exp_id = run_tag[len(LMAO_JOB_TYPE_PREFIX):].split("/")
        logger.info(f"Project name (Experiment ID): {project_id} ({exp_id})")

        # get details for the project and the run
        lmao_project = _lmao_stub.list_projects(ListProjectsRequest(
          workspace_id = workspace_id,
          project_id_or_name = project_id
        )).projects[0]
        lmao_run = _lmao_stub.get_run_details(Run(
          workspace_id = workspace_id,
          project_id = project_id,
          experiment_id = exp_id
        ))

        # create the experiment config and set all the values in the _lmaoConfig object that will be passed
        # to the LMAO class despite multiple initialisations, this ensures that when the user also uses LMAO
        # class all the values are already filled up.
        exp_config = ExperimentConfig.from_json(lmao_run.config)
        _lmaoConfig.set(
          project_name = lmao_project.project_name,
          project_id = lmao_run.project_id,
          experiment_id = exp_id,
          save_to_relic = exp_config.save_to_relic,
          enable_system_monitoring = exp_config.enable_system_monitoring,
          store_git_details = True,
        )
        logger.info("LMAO Config:\n" + pformat({
          "kv": _lmaoConfig.kv,
          "experiment": exp_config.to_dict()
        }, compact=True))
        args = ()
        kwargs = exp_config.run_kwargs

      else:
        # check if there is a specific relic for this job
        relic = Relics("cache", workspace_id)
        _in = f"{job_id}/args_kwargs"
        if run_tag:
          _in += f"_{run_tag}"
        logger.info(f"Looking for init.pkl at {_in}")
        if relic.relic is not None and relic.has(_in):
          (args, kwargs) = relic.get_object(_in)
        else:
          args, kwargs = (), {}

      # call the damn thing
      out = self.op(*args, **kwargs)

      # save the output to the relevant place, LMAO jobs are not saved to the relic
      if run_tag.startswith(LMAO_JOB_TYPE_PREFIX):
        logger.info("NBX-LMAO runs does not store function returns in Relics")
      else:
        _out = f"{job_id}/return"
        if run_tag:
          _out += f"_{run_tag}"
        logger.info(f"Saving output to {_out}")
        if relic.relic is not None:
          relic.put_object(_out, out)

      # last step mark as completed
      status = Job.Status.COMPLETED
    except Exception as e:
      U.log_traceback()
    finally:
      logger.info(f"Job {job_id} completed with status {status}")
      if hasattr(tracer, "job_proto"):
        tracer.job_proto.status = status
        tracer._rpc(f"RPC error in ending job {job_id}")
      U.hard_exit_program()

  def serve(self, host: str = "0.0.0.0", port: int = 8000, *, model_name: str = None):
    """Run a serving API endpoint"""

    # Unlike a run above where it is only going to be triggered once and all the metadata is already indexed
    # in the DB, this is not the case with deploy. But with deploy we can get away with something much simpler
    # which is using a more complicated ModelProto.metadata object
    # init_folder = U.env.NBOX_JOB_FOLDER("")
    # if not init_folder:
    #   raise RuntimeError("NBOX_JOB_FOLDER not set")
    # if not os.path.exists(init_folder):
    #   raise RuntimeError(f"NBOX_JOB_FOLDER {init_folder} does not exist")

    # fp_bin = U.join(init_folder, "model_proto.msg")
    # serving_type = SST.NBOX
    # if fp_bin:
    #   model_proto: ModelProto = read_file_to_binary(fp_bin, ModelProto())
    #   logger.info(model_proto)
    #   serving_type = model_proto.metadata.get("serving_type", SST.NBOX)

    try:
      serve_operator(
        op_or_app = self.op,
        # serving_type = serving_type,
        host = host,
        port = port,
        model_name = model_name
      )
    except Exception as e:
      U.log_traceback()
      logger.error(f"Failed to serve operator: {e}")
      U.hard_exit_program()
