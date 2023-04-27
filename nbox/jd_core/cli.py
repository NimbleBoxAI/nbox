from functools import partial

from nbox.jd_core.upload import upload_job_folder
from nbox.jd_core.jobs import get_job_list, Job
from nbox.jd_core.serving import print_serving_list, Serve

JobsCli = {
  "status": get_job_list,
  "upload": partial(upload_job_folder, "job")
}


ServeCli = {
  "status": print_serving_list,
  "upload": partial(upload_job_folder, "serving")
}
