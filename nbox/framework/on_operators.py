"""
Though initially this code just like the `on_ml.py <nbox.framework.on_ml.html>`_
code was build separately for each different package that we supported and that
led to just some really duct taping and hacking around that made it inefficient
to scale. The lesson is that premature attempts to scale are as bad as not being
able to scale. On that end eventually this file will be broken apart by packages
but hopefully it is in a way better. My depth limit = 2.

Read the code for best understanding.
"""

import inspect # used for __doc__
from functools import partial
from datetime import timedelta
from nbox.utils import isthere, logger


################################################################################
# Airflow
# =======
# Airflow is a great tool, but, read this
# https://nimblebox.notion.site/Control-Flow-vs-Data-Flow-332916c53b7f4a42a0b37d6aa365d4b7
################################################################################

class AirflowMixin:
  @isthere("airflow", soft = False)
  def nbx_job_to_airflow_operator(job, timeout: timedelta = None, operator_kwargs = {}):
    """
    Args:
      operator (Operator): nbox.Operator object to be converted to Airflow Operator
      timeout (timedelta, default=None): in how many seconds the operator should timeout
    """
    operator_kwargs_dict = {}
    operator_kwargs_dict["execution_timeout"] = timeout
    operator_kwargs_dict["sla"] = timeout
    operator_kwargs_dict["task_id"] = operator.__class__.__name__

    try:
      comms = operator.comms()
    except:
      comms = {}

    # this is currently assumed to be rst text because this is what we are using
    # for documentation at NBX
    doc = operator.__doc__ # this is in between class xxx and def __init__
    doc = doc if doc else ""
    init_doc = inspect.getdoc(operator.__init__) # this is doc for __init__
    init_doc = init_doc if init_doc else ""
    full_doc = doc + "\n" + init_doc
    full_doc = None if not full_doc else full_doc
    operator_kwargs_dict["doc_rst"] = full_doc

    # update the dict with information
    operator_kwargs_dict.update(comms)
    operator_kwargs_dict.update(operator_kwargs)

    # importing inside the function  
    from airflow.models.baseoperator import BaseOperator
    from ..init import reset_log
    reset_log()

    operator = BaseOperator(
      # items planned to be supported
      email = None,
      email_on_retry = False,
      email_on_failure = False,
      
      # documentation to be handled, currently assumed to be rst documentation
      doc = None,
      doc_md = None,
      doc_json = None,
      doc_yaml = None,

      # these can be the hooks that user defines for Operators
      # ie. is the airflow.operator executes user defines which of
      # their functions to call
      on_execute_callback = None,
      on_failure_callback = None,
      on_success_callback = None,
      on_retry_callback = None,

      # others are ignored as of now
      **operator_kwargs_dict
    )

    return operator


################################################################################
# Prefect
# =======
# https://github.com/PrefectHQ/prefect
################################################################################

class PrefectMixin:
  @classmethod
  def from_prefect_flow(operator_cls, dag):
    raise NotImplementedError
  
  def to_prefect_flow():
    raise NotImplementedError

  @classmethod
  def from_prefect_task(operator_cls, dag):
    raise NotImplementedError

  def to_prefect_task():
    raise NotImplementedError


