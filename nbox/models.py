"""
{% CallOut variant="warning" label="This is experimental and not ready for production use, can be removed without notice" /%}
"""

# bringing back the dead from the ashes
# this is a file that contains some pre built templates for the users to make their own training workflows

import json
from typing import Dict, Any, Union
from abc import ABCMeta, abstractmethod

from nbox.utils import logger
from nbox.projects import Project
from nbox.sublime.proto.lmao_v2_pb2 import Run as _Runpb


class ModelTrainer(metaclass = ABCMeta):
  def __init__(self, project_id: str):
    self.pid = project_id
    self.proj = None

  def init(self):
    self.proj = Project(id = self.pid)
    self.relic = self.proj.get_relic()

  # There's a couple of things that user needs to fill up in order for this to work smoothely
  # like how does th user want to train the model, or what happens when a user calls to make
  # a inference and saving and loading from checkpoints to quickly get to a certain artifacts

  @abstractmethod
  def train_model(self, *args, **kwargs):
    """
    Trains the model
    """

  @abstractmethod
  def infer(self, *args, **kwargs) -> Dict[str, Any]:
    """
    Runs the inference for any inputs that the user passes
    """

  @abstractmethod
  def save_checkpoint(self, *args, **kwargs) -> str:
    """
    Returns the path to the checkpoint
    """

  @abstractmethod
  def load_checkpoint(self, folder: str) -> None:
    """
    Loads the checkpoint and assigns the relevant attributes like model
    """

  # The next set of functions are provided by this framework to coordinate with NBX to get things
  # done correctly like actual saving of files on the Relics or deploying and running as workflow
  # or logging of metrics

  def log(self, log: Dict[str, Union[str, int, float]], step: int, **kwargs):
    """Logs the metrics to NBX"""
    if hasattr(self, "exp_tracker"):
      self.exp_tracker.log(log, step=step, **kwargs)
    elif hasattr(self, "live_tracker"):
      self.live_tracker.log(log)
    else:
      raise ValueError("No tracker found, initialize using train() or serve()")

  def checkpoint(self, *args, **kwargs):
    save_path = self.save_checkpoint(*args, **kwargs)
    if not type(save_path) == str:
      raise TypeError("Checkpoint path must be a string, save_checkpoint() returned a non string value")
    if not save_path:
      raise ValueError("Checkpoint path cannot be empty, save_checkpoint() returned an empty string")
    logger.info(f"Checkpoint created at {save_path}")
    self.exp_tracker.save_file(save_path)
    self.run_pb = self._update_run_config(
      latest_checkpoint = save_path
    )

  def get_checkpoint(self, experiment_id: str, name: str = ""):
    # if name is not provided we will get whatever is the latest checkpoint
    lmao_stub = self.proj.get_lmao_stub()
    run = lmao_stub.get_run_details(_Runpb(
      workspace_id = self.proj.workspace_id,
      project_id = self.pid,
      experiment_id = experiment_id,
    ))
    if not name:
      name = json.loads(run.config).get("latest_checkpoint", None)
      if name == None:
        raise ValueError("No checkpoint found")
    rp = f'{experiment_id}/{name}'
    logger.info(f"Loading checkpoint:\n  from: {rp}\n  to: {name}")
    self.relic.get_from(name, rp)
    self.load_checkpoint(name)

  def deploy(self, job_id: str):
    pass

  # all the methods below are helpers for the above methods

  def train(self, *args, **kwargs):
    """Run the training part of the pipeline"""
    if self.proj is None:
      self.init()
    self.exp_tracker = self.proj.get_exp_tracker()
    self.run_pb = self.exp_tracker.run
    self.train_model(*args, **kwargs)

  def serve(self, experiment_id: str, checkpoint: str = ""):
    """Run the inference part of the pipeline"""
    if self.proj is None:
      self.init()
    self.live_tracker = self.proj.get_live_tracker()
    self.serving_pb = self.live_tracker.serving
    self.get_checkpoint(experiment_id, checkpoint)

  def _update_run_config(self, **kwargs) -> _Runpb:
    """Updates the run config with the key and value"""
    run = self.run_pb
    cfg = json.loads(run.config)
    for k, v in kwargs.items():
      cfg[k] = v
    run.config = json.dumps(cfg)
    run.update_keys.append("config")
    ack = self.exp_tracker.lmao.update_run_status(run)
    if not ack.success:
      raise Exception(f"Failed to update run config: {ack.message}")
    run.update_keys.pop()
    return run
