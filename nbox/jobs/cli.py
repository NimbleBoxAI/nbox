import os
import jinja2
import subprocess
import webbrowser

from .utils import Subway
from ..utils import logger
from .. import utils as U
from ..init import nbox_session
from ..auth import secret

web_server_subway = Subway("https://nimblebox.ai", nbox_session)

################################################################################
# Jobs CLI
# ========
# This has functions for the CLI of NBX-Jobs
################################################################################

def new_job(project_name):
  import os, re
  from datetime import datetime
  created_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

  out = re.findall("^[a-zA-Z_]+$", project_name)
  if not out:
    raise ValueError("Project name can only contain letters and underscore")

  if os.path.exists(project_name):
    raise ValueError(f"Project {project_name} already exists")

  # ask user requirements here for customisation
  run_on_build = input("> Is this a run on build job? (y/N) ").lower() == "y"

  scheduled = None
  instance = None
  if run_on_build:
    # in this case the job will be run on a nbx-build instance (internal testing)
    # real advantage is ability to run on GPUs and persistent storage
    logger.info("This job will run on NBX-Build")
    instance = input("> Instance name or ID: ").strip()
  else:
    logger.info("This job will run on NBX-Jobs")
    scheduled = input("> Is this a recurring job (y/N)? ").lower() == "y"
    if scheduled:
      logger.info("This job will be scheduled to run on a recurring basis")
    else:
      logger.info(f"This job will run only once")
  
  logger.info(f"Creating a folder: {project_name}")
  os.mkdir(project_name)
  os.chdir(project_name)

  # jinja is cool
  assets = U.join(U.folder(U.folder(__file__)), "assets")
  path = U.join(assets, "job_new.jinja")
  with open(path, "r") as f, open("exe.py", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        run_on_build = run_on_build,
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
        instance = instance
    ))

  path = U.join(assets, "job_new_readme.jinja")
  with open(path, "r") as f, open("README.md", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
    ))

  open("requirements.txt", "w").close() # ~ touch requirements.txt

  logger.debug("Completed")

def deploy(folder):
  """Deploy a job on NimbleBox.ai's NBX-Jobs. This convinience function will
  just run ``./exe.py deploy``

  Args:
    folder (str, optional): Folder to deploy. Defaults to "./".
  """

  os.chdir(folder)
  subprocess.call(["python", "exe.py", "deploy"])

def open_jobs():
  webbrowser.open(secret.get("nbx_url")+"/"+"jobs")

