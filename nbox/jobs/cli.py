from .utils import Subway
from ..utils import nbox_session, folder, join
from ..auth import secret

import jinja2
import logging
import webbrowser
logger = logging.getLogger()
web_server_subway = Subway("https://nimblebox.ai", nbox_session)

################################################################################
# Jobs CLI
# ========
# This has functions for the CLI of NBX-Jobs
################################################################################

def new_job(project_name):
  logger.info("-" * 69)
  import os, re
  from datetime import datetime
  created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  out = re.findall("^[a-zA-Z_]+$", project_name)
  if not out:
    raise ValueError("Project name can only contain letters and underscore")

  if os.path.exists(project_name):
    raise ValueError(f"Project {project_name} already exists")

  # ask user requirements here for customisation
  scheduled = input("> Is this a recurring job (y/N)? ").lower() == "y"
  if scheduled:
    logger.info("This job will be scheduled to run on a recurring basis")
  else:
    logger.info(f"This job will run only once")
  
  logger.info(f"Creating a folder: {project_name}")
  os.mkdir(project_name)
  os.chdir(project_name)

  # jinja is cool
  path = join(folder(folder(__file__)), "assets", "job_new.jinja")
  with open(path, "r") as f, open("exe.py", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
    ))

  path = join(folder(folder(__file__)), "assets", "job_new_readme.jinja")
  with open(path, "r") as f, open("README.md", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
    ))

  open("requirements.txt", "w").close() # ~ touch requirements.txt

  logger.info("Completed")
  logger.info("-" * 69)

def deploy(folder):
  """Deploy a job on NimbleBox.ai's NBX-Jobs. This convinience function will
  just run ``./exe.py deploy``

  Args:
    folder (str, optional): Folder to deploy. Defaults to "./".
  """

  import os, subprocess

  print(os.path.abspath(folder))

  os.chdir(folder)
  subprocess.call(["python", "exe.py", "deploy"])

def open_jobs():
  webbrowser.open(secret.get("nbx_url")+"/"+"jobs")

