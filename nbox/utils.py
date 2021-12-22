# this file has bunch of functions that are used everywhere

import os
import io
import hashlib
import requests
import tempfile
import randomname
from PIL import Image
from datetime import timedelta
from types import SimpleNamespace
from time import time, sleep as _sleep

import logging

import numpy as np
import torch

# since nbox hsa become much bigger than waht 

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%d-%m-%y/%H:%M:%S",
)

nbox_session = requests.Session()

def _isthere(*packages):
    for package in packages:
        try:
            __import__(package)
        except Exception:
            return False
    return True

def isthere(*packages):
    def wrapper(fn):
        def _fn(*args, **kwargs):
            # since we are lazy evaluating this thing, we are checking when the function
            # is actually called. This allows checks not to happen during __init__.
            for package in packages:
                if not _isthere(package):
                    # raise a warning, let the modulenotfound exception bubble up
                    logging.warn(
                        f"{package} is not installed, but is required by {fn.__module__}, some functionality may not work"
                    )
            return fn(*args, **kwargs)
        return _fn
    return wrapper

# ----- functions


def fetch(url):
    # efficient loading of URLs
    fp = os.path.join(tempfile.gettempdir(), hash_(url))
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


def get_random_name():
    return randomname.generate()


def hash_(item, fn="md5"):
    return getattr(hashlib, fn)(str(item).encode("utf-8")).hexdigest()


def convert_to_list(x):
    # recursively convert tensors -> list
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return {k: convert_to_list(v) for k, v in x.items()}
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        x = np.nan_to_num(x, -1.42069)
        return x.tolist()
    else:
        raise Exception("Unknown type: {}".format(type(x)))


# --- classes

# # Console is a rich console wrapper for beautifying statuses
# class Console:
#     T = SimpleNamespace(
#         clk="deep_sky_blue1",  # timer
#         st="bold dark_cyan",  # status + print
#         fail="bold red",  # fail
#         inp="bold yellow",  # in-progress
#         nbx="bold bright_black",  # text with NBX at top and bottom
#         rule="dark_cyan",  # ruler at top and bottom
#         spinner="weather",  # status theme
#     )
# 
#     def __init__(self):
#         self.c = richConsole()
#         self._in_status = False
#         self.__reset()
# 
#     def rule(self, title: str):
#         self.c.rule(f"[{self.T.nbx}]{title}[/{self.T.nbx}]", style=self.T.rule)
# 
#     def __reset(self):
#         self.st = time()
# 
#     def __call__(self, x, *y):
#         cont = " ".join([str(x)] + [str(_y) for _y in y])
#         if not self._in_status:
#             self._log(cont)
#         else:
#             self._update(cont)
# 
#     def sleep(self, t: int):
#         for i in range(t):
#             self(f"Sleeping for {t-i}s ...")
#             _sleep(1)
# 
#     def _log(self, x, *y):
#         cont = " ".join([str(x)] + [str(_y) for _y in y])
#         t = str(timedelta(seconds=int(time() - self.st)))[2:]
#         self.c.print(f"[[{self.T.clk}]{t}[/{self.T.clk}]] {cont}")
# 
#     def start(self, x="", *y):
#         self.__reset()
#         cont = " ".join([str(x)] + [str(_y) for _y in y])
#         self.status = self.c.status(f"[{self.T.st}]{cont}[/{self.T.st}]", spinner=self.T.spinner)
#         self.status.start()
#         self._in_status = True
# 
#     def _update(self, x, *y):
#         t = str(timedelta(seconds=int(time() - self.st)))[2:]
#         cont = " ".join([str(x)] + [str(_y) for _y in y])
#         self.status.update(f"[[{self.T.clk}]{t}[/{self.T.clk}]] [{self.T.st}]{cont}[/{self.T.st}]")
# 
#     def stop(self, x):
#         self.status.stop()
#         del self.status
#         self._log(x)
#         self._in_status = False
