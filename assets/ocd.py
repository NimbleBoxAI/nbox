#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from time import sleep, time
from datetime import timedelta
from rich.console import Console

class T:
  clk = "deep_sky_blue1"   # timer
  st = "bold dark_cyan"    # status + print
  fail = "bold red"        # fail
  inp = "bold yellow"      # in-progress
  nbx = "bold grey0"       # text with NBX at top and bottom
  rule = "dark_cyan"       # ruler at top and bottom
  spinner = "moon"         # status theme

# looking at all the colors might be annoying, but it is simplest way to debug and ensure
# code is working as expected
console = Console()

# the process is step by step so perform sleep tasks inplace of .ocd() tasks
console.rule(f"[{T.nbx}]NBX-OCD[/{T.nbx}]", style = T.rule)
st = time()

# Step 1: Obtain the access token
with console.status("", spinner = T.spinner) as status:
  status.update(f"[{T.st}]Getting access tokens ...[/{T.st}]")
  sleep(2)
console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] Access token obtained")

# Step 2: Converting torch model to ONNX
with console.status("", spinner = T.spinner) as status:
  status.update(f"[{T.st}]Converting torch model to ONNX ...[/{T.st}]")
  sleep(3)
console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] ONNX model conversion complete")

# Step 3: Get one-time-url
with console.status("", spinner = T.spinner) as status:
  status.update(f"[{T.st}]Getting one-time-url ...[/{T.st}]")
  sleep(2)
console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] One-time-url obtained")

# Step 4: Upload file to s3
with console.status("", spinner = T.spinner) as status:
  status.update(f"[{T.st}]Uploading file to s3 ...[/{T.st}]")
  sleep(3)
console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] File uploaded to s3")

# Step 5: Updating Web-Server about the status of the upload
with console.status("", spinner = T.spinner) as status:
  status.update(f"[{T.st}]Updating Web-Server about the status of the upload[/{T.st}]")
  sleep(2)
console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] Web-Server informed")

# Step 6: Polling while the model is being deployed
with console.status(f"[{T.st}]Polling while the model is being deployed ...[/{T.st}]", spinner = T.spinner) as status:
  status.update(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] Current Status: [{T.inp}]conversion.in-progress")
  sleep(3)
  if random.random() > 0.2:
    console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] [{T.st}]Conversion successful")
    sleep(1)
    status.update(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] Current Status: [{T.inp}]deployment.in-progress")
    sleep(3)
    if random.random() > 0.2:
      console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] [{T.st}]Model is deployed at URL:")
      console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]]     https://depl.nbx.ai/yashbonde/its-britney-bish")
    else:
      status.update(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] Current Status: [{T.fail}]deployment.failed")
      console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] [{T.fail}]Model deployment failed")
  else:
    status.update(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] Current Status: [{T.fail}]conversion.failed")
    console.print(f"[[{T.clk}]{str(timedelta(seconds=int(time() - st)))[2:]}[/{T.clk}]] [{T.fail}]Model conversion failed")

console.rule(f"[{T.nbx}]NBX-OCD Complete[/{T.nbx}]", style = T.rule)