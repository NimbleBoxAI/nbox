"""
With NimbleBox you can run multi-cluster wide workloads from anywhere. This requires capabilities around distributed computing,
process management. The code here is tested along with `nbox.Relic` to perform distributed local and cloud processing.

{% CallOut variant="success" label="If you find yourself using this reach out to NimbleBox support." /%}
"""

import os

import nbox.utils as U
from nbox.utils import logger, lo
from nbox.relics import Relics
from nbox.operator import Operator
from nbox.auth import secret, AuthConfig
from nbox.nbxlib.tracer import Tracer
from nbox.hyperloop.jobs.job_pb2 import Job
from nbox.nbxlib.serving import serve_operator

from nbox.lmao import ExperimentConfig, LMAO_RM_PREFIX
from nbox.projects import Project, ProjectState

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

      # this is important since nbox uses AuthConfig.workspace_id place to get workspace_id from while the init_container
      # might place it at a different place. as of this writing, init_container -> "workspace_id" and nbox -> "config.global.workspace_id"
      secret.put(AuthConfig.workspace_id, workspace_id, True) 

      job_id = tracer.job_id
      self.op.propagate(_tracer = tracer)

      # get the user defined tag 
      run_tag = os.getenv("NBOX_RUN_METADATA", "")
      logger.info(f"Run Tag: {run_tag}")

      # in the NimbleBox system we provide tags for each key which essentially tells what is the behaviour
      # of the job. For example if it contains the string LMAO which means we need to initialise a couple
      # of things, or this can be any other job type
      args, kwargs = (), {}
      if run_tag.startswith(LMAO_RM_PREFIX):
        # originally we had a strategy to use Relics to store the information about the initialisation and passed args
        # however we are not removing that because we don't want to spend access money when we are anyways storing all
        # the information in the LMAO DB. so now we get the details of the run and get all the information from there.
        #
        # update (27/02/23): We completed the integration of LMAO with NBX-Projects to get beautifully simple interface
        #   for your entire MLOps pipeline. So copying the style from LMAO, we have something called ProjectState that
        #   has some variables that simplify the client side code when they don't have to pass any ids, it's all
        #   inferred.

        project_id, exp_id = run_tag[len(LMAO_RM_PREFIX):].split("/")
        logger.info(f"Project name (Experiment ID): {project_id} ({exp_id})")
        ProjectState.project_id = project_id
        ProjectState.experiment_id = exp_id

        # create the central project class and get the experiment tracker
        proj = Project()
        logger.info(lo("Project data:", **proj.data))
        exp_tracker = proj.get_exp_tracker()
        lmao_run = exp_tracker.run
        exp_config = ExperimentConfig.from_json(lmao_run.config)
        kwargs = exp_config.run_kwargs

      elif run_tag.startswith(RAW_DIST_RM_PREFIX):
        relic = Relics(RAW_DIST_RELIC_NAME, workspace_id)
        _pkl_id_in = run_tag[len(RAW_DIST_RM_PREFIX):] + "_in"
        logger.info(f"Looking for init.pkl at {_pkl_id_in}")
        (args, kwargs) = relic.get_object(_pkl_id_in)

      elif run_tag.startswith(SILK_RM_PREFIX):
        # 27/03/2023: We are adding a new job type called Silk
        trace_id = run_tag[len(SILK_RM_PREFIX):]
        logger.info(f"Running trace: {trace_id}")
        kwargs = {"trace_id": trace_id}

      # call the damn thing
      st = U.SimplerTimes.get_now_i64()
      out = self.op(*args, **kwargs)

      # save the output to the relevant place, LMAO jobs are not saved to the relic
      time_taken = U.SimplerTimes.get_now_i64() - st
      logger.info(lo(
        f"NBXLet: {run_tag} job completed, here's some stats:",
        time_taken = time_taken
      ))

      if run_tag.startswith(RAW_DIST_RM_PREFIX):
        _pkl_id_out = run_tag[len(RAW_DIST_RM_PREFIX):] + "_out"
        logger.info(f"Storing for output at {_pkl_id_out}")
        relic.put_object(_pkl_id_out, out)

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

    run_tag = os.getenv("NBOX_RUN_METADATA", "")
    logger.info(f"Run Tag: {run_tag}")

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

# Here's all the different tags that we use
# 
# Raw dist system uses Relics as an intermediate storage and jobs as compute nodes
# this is not the most optimal way but this is technology demonstration
RAW_DIST_RELIC_NAME = "tmp_cache"
RAW_DIST_RM_PREFIX = "NBXOperatorRawDist-"
RAW_DIST_ENV_VAR_PREFIX = "NBX_OperatorRawDist_"

# Silk is a production grade pipeline execution engine that can run any python code
SILK_RM_PREFIX = "ComputeSilk/"
