import os
import json
import requests
from getpass import getpass

from nbox.utils import join, Console


def get_access_token(nbx_home_url, username, password=None):
    password = getpass("Password: ") if password is None else password
    console = Console()
    console.start("Getting access tokens ...")
    try:
        r = requests.post(url=f"{nbx_home_url}/api/login", json={"username": username, "password": password})
    except Exception as e:
        raise Exception(f"Could not connect to NBX. You cannot use any cloud based tool!")

    if r.status_code == 401:
        console.stop("Invalid username/password")
        print("::" * 20 + " Invalid username/password. Please try again!")
        return False
    elif r.status_code == 200:
        access_packet = r.json()
        access_token = access_packet.get("access_token", None)
        console.stop("Access token obtained")
        return access_token
    else:
        console.stop(f"Unknown error: {r.status_code}")
        raise Exception(f"Unknown error: {r.status_code}")


def create_secret_file(username, access_token, nbx_url):
    folder = join(os.path.expanduser("~"), ".nbx")
    os.makedirs(folder, exist_ok=True)
    fp = join(folder, "secrets.json")
    with open(fp, "w") as f:
        f.write(json.dumps({"username": username, "access_token": access_token, "nbx_url": nbx_url}))


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
            try:
                self.secrets = json.loads(
                    requests.get("https://raw.githubusercontent.com/NimbleBoxAI/nbox/master/assets/sample_config.json").content.decode(
                        "utf-8"
                    )
                )
            except Exception as e:
                raise Exception(f"Could not connect to NBX. You cannot use any cloud based tool!")

            # populate with the first time things
            nbx_home_url = "https://www.nimblebox.ai"
            username = input("Username: ")
            access_token = None
            while not access_token:
                access_token = get_access_token(nbx_home_url, username)
                self.secrets["access_token"] = access_token
            self.secrets["username"] = username
            self.secrets["nbx_url"] = nbx_home_url
            self.save()
        else:
            with open(self.fp, "r") as f:
                self.secrets = json.load(f)
            print("Successfully loaded secrets!")

    def __repr__(self):
        return json.dumps(self.secrets, indent=2)

    def get(self, item):
        return self.secrets[item]

    def save(self):
        with open(self.fp, "w") as f:
            f.write(json.dumps(self.secrets, indent=2))

    def add_ocd(self, model_id, url, nbox_meta, access_key):
        if "ocd" not in self.secrets:
            return
        ocd = list(filter(lambda x: x["url"] == url, self.secrets["ocd"]))
        if ocd:
            # no need to udpate if already present
            return
        self.secrets["ocd"].append(
            {
                "url": url,
                "model_id": model_id,
                "nbox_meta": nbox_meta,
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
            # raise IndexError(f"URL {url} not found in secrets.json")
            self.add_ocd(None, url, None, self.nbx_api_key)
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


def reinit_secret():
    global secret
    secret = Secrets()


if os.getenv("NBX_AUTH", False):
    secret = None
else:
    secret = Secrets()
