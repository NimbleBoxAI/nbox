from nbox import Operator, logger
from nbox.lib.shell import ShellCommand, GitClone

class Intro(Operator):
  # yet another time waster
  def __init__(self):
    super().__init__()

  def forward(self):
    logger.info("Welcome to NimbleBox!")


class ThingsShellCanDo(Operator):
  def __init__(self) -> None:
    super().__init__()
    self.amar = ShellCommand("echo 'Lana Del Ray'")
    self.akbar = ShellCommand("echo 'Run any shell command, like read files'")
    self.anthony = ShellCommand("echo 'Kenrik'")
  
  def forward(self):
    self.amar()
    self.akbar()
    self.anthony()


class Magic(Operator):
  # named after the Dr.Yellow train that maintains the Shinkansen
  def __init__(self, strength = "very"):
    super().__init__()
    self.strength = strength
    self.intro = Intro()
    self.shell_demo = ThingsShellCanDo()
    # self.cloner = GitClone("https://github.com/yashbonde/yQL")

  def forward(self):
    # self.cond1(cond = 1)
    self.intro()
    self.shell_demo()
    # self.cloner()


class MagicServing(Operator):
  def __init__(self):
    super().__init__()
    self.intro = Intro()
    self.cities = [
      "Mumbai",
      "Delhi",
      "Banares",
      "Bangalore",
      "Chennai",
      "Kolkata",
      "Hyderabad",
      "Ahmedabad",
      "Pune",
      "Jaipur",
      "Surat",
      "Lucknow",
      "Kanpur",
      "Nagpur",
      "Patna",
      "Indore",
      "Vadodara",
      "Agra",
      "Siliguri",
      "Coimbatore",
    ]

  def __remote_init__(self):
    logger.info("Running remote init: it will only run at initialisation")
    logger.info("So you can do things like:")
    logger.info("    * initialise your model")
    logger.info("    * connect to databases")
    logger.info("    * create games!")
    self.intro()

  def forward(self, x: str = ""):
    import random
    if not x:
      return {"data": f"Hi, I am {random.choice(self.cities)}!"}
    else:
      return {"data": f"Echo back: {x}!"}
