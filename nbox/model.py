"""
# NBX-Model

This will combine all the NimbleBox modules into a single callable.
"""


class Model:
  def __init__(
    self,
    # model: Any,
    # method: str = None,
    # pre: callable = None,
    # post: callable = None,
    # model_spec: ModelSpec = None,
    # verbose: bool = False
  ):
    """Top of the stack Model class."""
    raise NotImplementedError("WIP check back later.")

  def __repr__(self):
    return f"<nbox.Model: {self.model} >"

  def __call__(self, input_object):
    r"""Call is the most important UI/UX. The ``input_object`` can be anything from
    a tensor, an image file, filepath as string, string and is processed by ``pre`` function."""
    raise NotImplementedError("WIP check back later.")

  @classmethod
  def deserialise(cls):
    """Load ``ModelSpec`` and ``folder`` with the files in it and return a ``Model`` object."""
    raise NotImplementedError("WIP check back later.")


  def deploy(self,):
    """Serve your model on NBX-Deploy `read more <https://nimbleboxai.github.io/nbox/nbox.model.html>`_"""
    raise NotImplementedError("WIP check back later.")

  @staticmethod
  def train_on_instance():
    """Train this model on an NBX-Build Instance. Though this function is generic enough to execute
    any arbitrary code, this is built primarily for internal use."""
    raise NotImplementedError("WIP check nbox.Job")

  @staticmethod
  def train_on_job():
    """Train this model on NBX-Jobs. This same experience can be given by creating a new job
    that then can then be populated how we currently create a new job using NBX-Jobs CLI"""
    raise NotImplementedError("WIP check nbox.Job")

