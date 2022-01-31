"""
Jobs
====

``nbox.Job`` is a wrapper to the APIs that's it.
"""

import requests

from logging import getLogger
logger = getLogger()

from ..auth import secret


class Job:
  def __init__(self, id):
    self.id = id
    self.update()

  @classmethod
  def create(cls, name):
    # create new job from API
    r = requests.post(secret.get("url") + "/api/v1/create", json={"name": name})
    if not r.status_code == 200:
      raise Exception(f"Failed to create job {name} | {r.content.decode()}")

  def change_status(self, new_status):
    # nbox should only request and server should check if possible or not
    pass

  def update(self):
    # this function will update all information for the job
    r = requests.get(secret.get("url") + "/jobs/" + id, headers=secret.get("headers"))
    if not r.status_code == 200:
      raise Exception(f"Failed to load job {id} | {r.content.decode()}")

    data = r.json()
    self.name = data['name']
    self.created_at = data['created_at']
    self.code = data['code']
    self.auth_info = data['auth_info']
    self.schedule = data['schedule']
    self.dag = data['dag']
    self.status = data['status']
