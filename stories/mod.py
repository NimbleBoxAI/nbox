import os
os.environ["NBOX_LOG_LEVEL"] = "info"

import json
import fire
from time import sleep, time

from nbox import Job
from nbox.jd_core import print_job_list, new_job, upload_job_folder
from nbox.utils import SimplerTimes

from utils import hr

class JobsModule:
  def story(self, job_name: str = f"nboxS-{SimplerTimes.get_now_i64()}", n: int = 12):
    _st = time()
    hr(f"nbx jobs new --job_name='{job_name}'"); job = new_job(name=job_name, description="This is a test job"); print("JOB:", job)
    hr(f"nbx jobs pick --job_id='{job.id}' job_proto"); job = Job(job_id=job.id); print("STATUS:", job.status); print("CODE:", job.job_pb.code)
    hr(f"nbx jobs upload test:main --id '{job.id}'"); upload_job_folder("job", "test:main", id = job.id)
    hr(f"nbx jobs pick --job_id='{job.id}' job_proto"); job = Job(job_id=job.id); print("STATUS:", job.status); print("CODE:", job.job_pb.code)
    hr(f"nbx jobs --id '{job.id}' trigger"); job.trigger(); print("Sleeping for 2 seconds"); sleep(2)
    hr(f"nbx jobs --id '{job.id}' get_runs"); runs = job.get_runs(); print("RUNS:", len(runs))
    rid = runs[0]["run_id"]
    st = time()
    for i in range(n):
      runs = job.get_runs(); print(f"[{i:02d}/{n}]", "STATUS:", rid, runs[0]["status"])
      if runs[0]["status"] == "COMPLETED":
        break
      sleep(5)
    print("TOOK (seconds):", time() - st)

    # ---
    sleep(2)
    hr(f"nbx jobs pick --job_id '{job.id}' get_run_log '{rid}' - logs")
    out = Job(job_id=job.id).get_run_log(rid)
    logs = [json.dumps(x) for x in out["logs"]][:10]
    print("LOG ITEMS:", len(logs))
    print("LOGS:\n", "\n".join(logs))

    # ---
    hr(f"nbx jobs pick --job_id='{job.id}' delete"); Job(job_id=job.id).delete()
    hr(f"Full Time: {time() - _st} seconds", "+")

if __name__ == "__main__":
  fire.Fire({
    "jobs": JobsModule
  })