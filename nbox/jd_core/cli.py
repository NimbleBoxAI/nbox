from functools import partial

from nbox.jd_core.upload import upload_job_folder
from nbox.jd_core.jobs import print_job_list, Job, new_job, get_job_list
from nbox.jd_core.serving import print_serving_list, Serve

JobsCli = {
  "status": print_job_list,
  "list": print_job_list,
  "upload": partial(upload_job_folder, "job"),
  "pick": Job,
  "new": new_job,
}


ServeCli = {
  "status": print_serving_list,
  "upload": partial(upload_job_folder, "serving"),
  "pick": Serve,
}
