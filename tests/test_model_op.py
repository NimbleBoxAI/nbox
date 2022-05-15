from nbox.model import ModelOp

class ForecastingModel(ModelOp):
  def __init__(
    self,
    run_name: str,
  ):
    super().__init__(*args, **kwargs)
