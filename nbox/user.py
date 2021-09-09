from logging import warning
import os
import re
import json
import requests
from getpass import getpass

from nbox.utils import join, Console

def get_access_token(nbx_home_url, username, password=None):
    password = getpass("Password: ") if password is None else password
    console = Console()
    console.start("Getting access tokens ...")
    r = requests.post(
        url=f"{nbx_home_url}/api/login",
        json={"username": username, "password": password},
        verify=False,
    )
    if r.status_code == 401:
        console.stop("Invalid username/password")
        print("::" * 20 + " Invalid username/password. Please try again!")
    elif r.status_code == 200:
        access_packet = r.json()
        access_token = access_packet.get("access_token", None)
        console.stop("Access token obtained")
        return access_token
    else:
        console.stop(f"Unknown error: {r.status_code}")
        raise Exception(f"Unknown error: {r.status_code}")


class Secrets:
    # this is the user local store
    def __init__(self):
        # get the secrets file
        folder = join(os.path.expanduser("~"), ".nbx")
        os.makedirs(folder, exist_ok=True)
        self.fp = join(folder, "secrets.json")

        # if this is the first time starting this then get things from the nbx-hq
        if not os.path.exists(self.fp):
            # get the secrets JSON
            self.secrets = json.loads(
                re.sub(
                    r"//.*",
                    "",
                    requests.get("https://raw.githubusercontent.com/NimbleBoxAI/nbox/cloud-infer/assets/sample_config.json").content.decode(
                        "utf-8"
                    ),
                )
            )

            # populate with the first time things
            nbx_home_url = input("NimbleBox.ai home URL (default: https://www.nimblebox.ai/): ")
            if not nbx_home_url.strip():
                nbx_home_url = "https://www.nimblebox.ai/"

            username = input("Username: ")
            access_token = None
            while access_token is None:
                access_token = get_access_token(nbx_home_url, username)
                self.secrets["access_token"] = access_token
            self.secrets["username"] = username
            self.secrets["nbx_url"] = nbx_home_url
            self.save()
        else:
            with open(self.fp, "r") as f:
                self.secrets = json.loads(re.sub(r"//.*", "", f.read()))

    def __repr__(self):
        return json.dumps(self.secrets, indent=2)

    def get(self, item):
        return self.secrets[item]

    def save(self):
        with open(self.fp, "w") as f:
            f.write(json.dumps(self.secrets, indent=2))

    def add_ocd(self, model_id, url, nbx_meta, access_key):
        self.secrets["ocd"].append(
            {
                "url": url,
                "model_id": model_id,
                "nbx_meta": nbx_meta,
                "access_key": access_key,
                "api_hits": 0,
                "bytes_recieved": 0,
                "bytes_sent": 0,
                "bytes_total": 0,
            }
        )
        self.save()

    def update_ocd(self, url, bytes_recieved, bytes_sent):
        ocd = list(filter(lambda x: x["url"] == url, self.secrets["ocd"]))
        if not ocd:
            raise IndexError(f"URL {url} not found in secrets.json")
        ocd = ocd[0]
        ocd["bytes_recieved"] += bytes_recieved
        ocd["bytes_sent"] += bytes_sent
        ocd["bytes_total"] += bytes_sent + bytes_recieved
        ocd["api_hits"] += 1
        self.save()

    def get_ocd(self, url):
        ocd = list(filter(lambda x: x["url"] == url, self.secrets["ocd"]))
        if not ocd:
            raise IndexError(f"URL {url} not found in secrets.json")
        ocd = ocd[0]
        return ocd

    def update_access_token(self, access_token):
        self.secrets["access_token"] = access_token
        self.save()

    def update_username(self, username):
        self.secrets["username"] = username
        self.save()

# secret = Secrets()
