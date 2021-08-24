#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from time import sleep, time
from datetime import timedelta
from rich.console import Console

# looking at all the colors might be annoying, but it is simplest way to debug and ensure
# code is working as expected
console = Console()

# the process is step by step so perform sleep tasks inplace of .ocd() tasks
console.rule("[bold grey0]NBX-OCD[/bold grey0]", style = "dark_cyan")
st = time()

# Step 1: Obtain the access token
with console.status("", spinner = "moon") as status:
  status.update(f"[bold dark_cyan]Getting access tokens ...[/bold dark_cyan]")
  sleep(2)
console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] Access token obtained")

# Step 2: Converting torch model to ONNX
with console.status("", spinner = "moon") as status:
  status.update(f"[bold dark_cyan]Converting torch model to ONNX ...[/bold dark_cyan]")
  sleep(3)
console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] ONNX model conversion complete")

# Step 3: Get one-time-url
with console.status("", spinner = "moon") as status:
  status.update(f"[bold dark_cyan]Getting one-time-url ...[/bold dark_cyan]")
  sleep(2)
console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] One-time-url obtained")

# Step 4: Upload file to s3
with console.status("", spinner = "moon") as status:
  status.update(f"[bold dark_cyan]Uploading file to s3 ...[/bold dark_cyan]")
  sleep(3)
console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] File uploaded to s3")

# Step 5: Updating Web-Server about the status of the upload
with console.status("", spinner = "moon") as status:
  status.update(f"[bold dark_cyan]Updating Web-Server about the status of the upload[/bold dark_cyan]")
  sleep(2)
console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] Web-Server informed")

# Step 6: Polling while the model is being deployed
with console.status("[bold dark_cyan]Polling while the model is being deployed ...[/bold dark_cyan]", spinner = "moon") as status:
  status.update(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] Current Status: [bold yellow][code]conversion.in-progress")
  sleep(3)
  if random.random() > 0.0:
    console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] [bold dark_cyan]Conversion successful")
    sleep(1)
    status.update(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] Current Status: [bold yellow][code]deployment.in-progress")
    sleep(3)
    if random.random() > 0.0:
      console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] [bold dark_cyan]Model is deployed at URL:")
      console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]]     https://depl.nbx.ai/yashbonde/its-britney-bish")
    else:
      status.update(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] Current Status: [bold yellow][code]deployment.failed")
      console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] [bold red]Model deployment failed")
  else:
    status.update(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] Current Status: [bold red][code]conversion.failed")
    console.print(f"[[deep_sky_blue1]{str(timedelta(seconds=int(time() - st)))[2:]}[/deep_sky_blue1]] [bold red]Model conversion failed")

console.rule("[bold grey0]NBX-OCD Complete[/bold grey0]", style = "dark_cyan")
