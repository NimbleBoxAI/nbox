# ssympple parsing

class Mux():
  @staticmethod
  def process_list(x):
    # type checking/
    t0 = type(x[0])
    if t0 == list:
      raise ValueError("Mux does not support nested lists")
    if any([type(x_) != t0 for x_ in x]):
      raise ValueError("Mux does not support mixed types")
    
    # logic/
    if isinstance(t0, dict):
      x = {k: Mux.process_list([x_[k] for x_ in x]) for k in x[0].keys()}
    else:
      x = Mux.primitive(x)
    
    return x
  
  @staticmethod
  def process_dict(x):
    for k, v in x.items():
      if isinstance(v, dict):
        x[k] = Mux.process_dict(v)
      elif isinstance(v, list):
        x[k] = Mux.process_list(v)
      else:
        x[k] = Mux.primitive(v)
    return x

  @staticmethod
  def parse(x, *a, **b):
    if isinstance(x, dict):
      return Mux.process_dict(x, *a, **b)
    elif isinstance(x, list):
      return Mux.process_list(x, *a, **b)
    else:
      return Mux.primitive(x, *a, **b)

  def primitive(x):
    pass
