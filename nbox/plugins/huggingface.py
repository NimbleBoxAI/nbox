"""
This file contains the code for all the compatibility with the [huggingface platform](https://huggingface.co/)
"""

from nbox import logger, lo, Project
from nbox.utils import SimplerTimes
from nbox.plugins.base import import_error

try:
  import huggingface_hub as hfhub
  from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
  from transformers.training_args import TrainingArguments
except ImportError:
  raise import_error("transformers")


class NimbleBoxTrainerCallback(TrainerCallback):
  """
  This `TrainerCallback` talks to the [NimbleBox server](https://nimblebox.ai) to log training metrics and checkpoints.

  This will automatically detect if you are running on NimbleBox pods and will automatically pick the appropriate
  values from the environment variables such as your project ID, Run ID, etc. If you are running this outside of the
  NimbleBox, you can read more in our [docs](https://nimblebox.ai/docs).
  """

  def __init__(
    self,
    on_init_end: bool = True,
    on_train_begin: bool = True,
    on_train_end: bool = True,
    on_epoch_begin: bool = False,
    on_epoch_end: bool = False,
    on_step_begin: bool = False,
    on_substep_end: bool = False,
    on_step_end: bool = False,
    on_evaluate: bool = True,
    on_predict: bool = False,
    on_save: bool = True,
    on_log: bool = True,
    on_prediction_step: bool = False,
    *,
    debug: bool = False,
  ) -> None:
    self._initialized = False
    self.debug = debug
    self._on_init_end = on_init_end
    self._on_train_begin = on_train_begin
    self._on_train_end = on_train_end
    self._on_epoch_begin = on_epoch_begin
    self._on_epoch_end = on_epoch_end
    self._on_step_begin = on_step_begin
    self._on_substep_end = on_substep_end
    self._on_step_end = on_step_end
    self._on_evaluate = on_evaluate
    self._on_predict = on_predict
    self._on_save = on_save
    self._on_log = on_log
    self._on_prediction_step = on_prediction_step

  def setup(self, args, state, model, tokenizer, **kwargs):
    if self._initialized:
      return
    if state.is_world_process_zero:
      self.p = Project()

  def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_init_end):
      return
    logger.info("on_init_end: initialization of the Trainer is complete. Will load all the training arguments.")
    metadata = {
      "hf_training_args": args.to_dict()
    }
    self._tracker = self.p.get_exp_tracker(metadata = metadata)

  def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_train_begin):
      return
    logger.debug("on_train_begin: training is about to begin.")

  def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_train_end):
      return
    logger.debug("on_train_end: training is complete.")
    self._tracker.end()

  def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_epoch_begin):
      return
    logger.debug("on_epoch_begin: epoch is about to begin.")

  def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_epoch_end):
      return
    logger.debug("on_epoch_end: epoch is complete.")

  def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_step_begin):
      return
    logger.debug("on_step_begin: step is about to begin.")

  def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_substep_end):
      return
    logger.debug("on_substep_end: substep is complete.")

  def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_step_end):
      return
    logger.debug("on_step_end: step is complete.")

  def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_evaluate):
      return
    logger.debug("on_evaluate: evaluation is complete.")

  def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
    if not (state.is_world_process_zero or self._on_predict):
      return
    logger.debug("on_predict: prediction is complete.")

  def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_save):
      return
    logger.debug("on_save: checkpoint is saved.")

  def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, tokenizer=None, logs=None, **kwargs):
    if not (state.is_world_process_zero or self._on_log):
      return
    if logs is not None:
      logger.debug("on_log: logging is complete.")
      self._tracker.log(logs)

  def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    if not (state.is_world_process_zero or self._on_prediction_step):
      return
    logger.debug("on_prediction_step")
