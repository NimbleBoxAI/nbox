# from https://gist.github.com/yashbonde/62df9d16858a43775c22a6af00a8d707

import os
import io
import re
import json
import hashlib
import requests
import tempfile
from PIL import Image
from time import time
from datetime import timedelta
from rich.console import Console
from types import SimpleNamespace

import logging

# ----- functions

logging.basicConfig(level="INFO")


def info(x, *args):
    # because logging.info requires formatted strings
    x = repr(x)
    x = " ".join([x] + [repr(y) for y in args])
    logging.info(x)


def fetch(url):
    # efficient loading of URLs
    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        dat = requests.get(url).content
        with open(fp + ".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp + ".tmp", fp)
    return dat


def get_image(file_path_or_url):
    if os.path.exists(file_path_or_url):
        return Image.open(file_path_or_url)
    else:
        return Image.open(io.BytesIO(fetch(file_path_or_url)))


def folder(x):
    # get the folder of this file path
    return os.path.split(os.path.abspath(x))[0]


def join(x, *args):
    return os.path.join(x, *args)


def is_available(package: str):
    import importlib

    spam_spec = importlib.util.find_spec(package)
    return spam_spec is not None


def get_random_name():
    import randomname

    return randomname.generate()


def hash_(item, fn="md5"):
    return getattr(hashlib, fn)(str(item).encode("utf-8")).hexdigest()


# --- classes

# OCDConsole is a rich console wrapper for beautifying statuses
class OCDConsole:
    T = SimpleNamespace(
        clk="deep_sky_blue1",  # timer
        st="bold dark_cyan",  # status + print
        fail="bold red",  # fail
        inp="bold yellow",  # in-progress
        nbx="bold bright_black",  # text with NBX at top and bottom
        rule="dark_cyan",  # ruler at top and bottom
        spinner="weather",  # status theme
    )

    def __init__(self):
        self.c = Console()
        self.st = time()
        self._in_status = False

    def rule(self):
        self.c.rule(f"[{self.T.nbx}]NBX-OCD[/{self.T.nbx}]", style=self.T.rule)

    def __call__(self, x, *y):
        cont = " ".join([str(x)] + [str(_y) for _y in y])
        if not self._in_status:
            self._log(cont)
        else:
            self._update(cont)

    def _log(self, x, *y):
        cont = " ".join([str(x)] + [str(_y) for _y in y])
        t = str(timedelta(seconds=int(time() - self.st)))[2:]
        self.c.print(f"[[{self.T.clk}]{t}[/{self.T.clk}]] {cont}")

    def start(self, x="", *y):
        cont = " ".join([str(x)] + [str(_y) for _y in y])
        self.status = self.c.status(f"[{self.T.st}]{cont}[/{self.T.st}]", spinner=self.T.spinner)
        self.status.start()
        self._in_status = True

    def _update(self, x, *y):
        t = str(timedelta(seconds=int(time() - self.st)))[2:]
        cont = " ".join([str(x)] + [str(_y) for _y in y])
        self.status.update(f"[[{self.T.clk}]{t}[/{self.T.clk}]] [{self.T.st}]{cont}[/{self.T.st}]")

    def stop(self, x):
        self.status.stop()
        del self.status
        self._log(x)
        self._in_status = False


class Secrets:
    # this class is used to manage all auth related secrets by reading them from a file
    # and writing them back when the program exits
    def __init__(self, file_path=None):
        self.fp = file_path or join("/", "secrets.json")
        self.secrets = {}
        with open(file_path, "r") as f:
            # read the JSON file, remove comments and then load it
            self.secrets = json.loads(re.sub(r"//.*", "", f.read()))

    def __getattribute__(self, name: str):
        return self.get(name)

    def __getitem__(self, name: str):
        return self.get(name)

    def get(self, name: str):
        return self.secrets[name]
