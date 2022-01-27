# this file has bunch of functions that are used everywhere

import os
import io
import hashlib
import requests
import tempfile
import randomname
from uuid import uuid4

import logging


# common/
nbox_session = requests.Session()
# /common

# logging/
logger = logging.getLogger()
# /logging


# lazy_loading/

def isthere(*packages, soft = True):
    def wrapper(fn):
        def _fn(*args, **kwargs):
            # since we are lazy evaluating this thing, we are checking when the function
            # is actually called. This allows checks not to happen during __init__.
            for package in packages:
                try:
                    __import__(package)
                except ImportError:
                    if not soft:
                        raise Exception(f"{package} is not installed")
                    # raise a warning, let the modulenotfound exception bubble up
                    logger.warn(
                        f"{package} is not installed, but is required by {fn.__module__}, some functionality may not work"
                    )
            return fn(*args, **kwargs)
        return _fn
    return wrapper

def _isthere(*packages):
    for package in packages:
        try:
            __import__(package)
        except Exception:
            return False
    return True

# /lazy_loading

# file path/reading

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


def folder(x):
    # get the folder of this file path
    return os.path.split(os.path.abspath(x))[0]

def join(x, *args):
    return os.path.join(x, *args)

NBOX_HOME_DIR = join(os.path.expanduser("~"), ".nbx")

# /path

# misc/

def get_random_name(uuid = False):
    if uuid:
        return str(uuid4())
    return randomname.generate()

def hash_(item, fn="md5"):
    return getattr(hashlib, fn)(str(item).encode("utf-8")).hexdigest()

# /misc

# model/

def get_image(file_path_or_url):
    if os.path.exists(file_path_or_url):
        return Image.open(file_path_or_url)
    else:
        return Image.open(io.BytesIO(fetch(file_path_or_url)))

def convert_to_list(x):
    # recursively convert tensors -> list
    import torch
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return {k: convert_to_list(v) for k, v in x.items()}
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        x = np.nan_to_num(x, -1.42069)
        return x.tolist()
    else:
        raise Exception("Unknown type: {}".format(type(x)))

# /model

# pool/
from concurrent.futures import ThreadPoolExecutor, as_completed
POOL_SUPPORTED_MODES = ["thread"]

class Pool:
    def __init__(self, mode = "thread", max_workers = 2, _name: str = get_random_name(True)):
        """Threading is hard, your brain is not wired to handle parallelism. You are a blocking
        python program. So a blocking function for you.

        Args:
            mode (str, optional): There can be multiple pooling strategies across cores, threads,
                k8s, nbx-instances etc.
            max_workers (int, optional): Numbers of workers to use
            _name (str, optional): Name of the pool, used for logging

        Usage:
            
            def sleep_and_return(x):
                from time import sleep
                sleep(x)
                return x
            
            pool = Pool()
            results = pool(sleep_and_return, (2,),(6,),(4,),(5,)) # inputs must be a tuple
        """
        if mode not in POOL_SUPPORTED_MODES:
            raise Exception(f"Only {', '.join(POOL_SUPPORTED_MODES)} mode(s) are supported")

        self.mode = mode
        self.pool = None
        logger.info(f"Starting ThreadPool ({_name}) with {max_workers} workers")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=_name
        )
        self.item_id = -1 # because +1 later
        self.futures = {}

    def __call__(self, fn, *args):
        """Run any function ``fn`` in parallel, where each argument is a list of arguments to
        pass to ``fn``. Result is returned in the same order as the input.
            
            thread(fn, a) for a in args -> list of results
        """
        assert callable(fn)
        assert isinstance(args[0], (tuple, list))

        futures = {}
        for i, x in enumerate(args):
            futures[self.executor.submit(fn, *x)] = i # insertion index

        self.item_id += len(futures)
        results = {}
        for future in as_completed(futures):
            try:
                result = future.result()
                results[futures[future]] = result # update that index
            except Exception as e:
                logger.error(f"{self.mode} error: {e}")
                raise e

        return [results[x] for x in range(len(results))]

# /pool


# --- classes

# this needs to be redone
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
