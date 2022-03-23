"""
This file provides support for the SSH tunneling into NBX-Build Instances.
"""
# If you have come till here, and you want to work on more things like this, reach out:
# research@nimblebox.ai

import os
import ssl
import socket
import certifi
import threading
from typing import List
from datetime import datetime
from functools import partial

from ..utils import NBOX_HOME_DIR, logger as nbx_logger
from .. import utils as U
from ..auth import secret
from ..instance import Instance

class FileLogger:
  def __init__(self, filepath):
    self.filepath = filepath
    self.f = open(filepath, "w")

    self.debug = partial(self.log, level="debug",)
    self.info = partial(self.log, level="info",)
    self.warning = partial(self.log, level="warning",)
    self.error = partial(self.log, level="error",)
    self.critical = partial(self.log, level="critical",)

  def log(self, message, level):
    self.f.write(f"[{datetime.utcnow().isoformat()}] {level}: {message}\n")
    self.f.flush()


class RSockClient:
  """
  This is a RSockClient. It handels the client socket where client is the user application trying to connect to "client_port"
  Connects to RSockServer listening on localhost:886.
  RSockServer recieves instructions as a string and sends back a response.
  RSockServer requires following steps to setup
  First, authentication:
    - Authentication happens by sending
      `"AUTH~{AUTH_TOKEN}"`
    - AUTH_TOKEN is not defined and is default to 'password'
  Seting config:
    - you can set config by sending
      `"SET_CONFIG~{instance}~{instance_port}"`
    - "instance" - Currently is the internal ip of the instance.
    - "instance_port" - What port users wants to connect to.
  Connect:
    - This Starts the main loop which
      1. Listen on client_port
      2. On connection, 
        a. Send AUTH
        b. If AUTH is successful, send SET_CONFIG
        c. If SET_CONFIG is successful, send CONNECT
        d. If CONNECT is successful, start io_copy
  IO_COPY:
    - This is the main loop that handles the data transfer between client and server. This is done by creating a new thread for each connection.
    - The thread is created by calling the function "io_copy" for each connection that is "server" and "client".
    - When a connection is closed, the loop is stopped.
  """

  def __init__(self, connection_id, client_socket, instance, instance_port, auth, logger: FileLogger, secure=False,):
    """
    Initializes the client.
    Args:
      client_socket: The socket that the client is connected to.
      instance: The instance that the client wants to connect to.
      instance_port: The port that the instance is listening on.
      auth: The authentication token that the client has to provide to connect to the RSockServer.
      secure: Whether or not the client is using SSL.
    
    """
    self.connection_id = connection_id
    self.client_socket = client_socket
    self.instance = instance
    self.instance_port = instance_port
    self.auth = auth
    self.secure = secure
    self.logger = logger

    self.client_auth = False
    self.rsock_thread_running = False
    self.client_thread_running = False

    self.logger.info('Starting client')
    self.connect_to_rsock_server()
    self.logger.info('Connected to RSockServer')
    self.authenticate()
    self.logger.info('Authenticated client')
    self.set_config()
    self.logger.info('Client init complete')

  def connect_to_rsock_server(self):
    """
    Connects to RSockServer.
    """
    self.logger.debug('Connecting to RSockServer')
    rsock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rsock_socket.connect(('rsocks.nimblebox.ai', 886))

    if self.secure:
      self.logger.info('Starting SSL')
      self.rsock_socket = ssl.wrap_socket(rsock_socket, ca_certs=certifi.where(), cert_reqs=ssl.CERT_REQUIRED)
    else:
      self.rsock_socket = rsock_socket

  def authenticate(self):
    """
    Authenticates the client.
    Sends `"AUTH~{AUTH_TOKEN}"` to RSockServer.
    """
    self.logger.info('Authenticating client')
    self.rsock_socket.sendall(bytes(f'AUTH~{self.auth}', 'utf-8'))
    auth = self.rsock_socket.recv(1024)
    auth = auth.decode('utf-8')
    if auth == 'OK':
      self.logger.info('Client authenticated')
    else:
      self.logger.erro('Client authentication failed')
      self.client_auth = False
      exit(1)
  
  def set_config(self):
    """
    Sets the config of the client.
    Sends `"SET_CONFIG~{instance}~{instance_port}"` to RSockServer.
    """
    self.logger.info('Setting config')
    self.rsock_socket.sendall(bytes(f'SET_CLIENT~{self.instance}~{self.instance_port}', 'utf-8'))
    config = self.rsock_socket.recv(1024)
    config = config.decode('utf-8')
    self.logger.info('Config set to {}'.format(config))
    if config == 'OK':
      self.logger.info('Config set')
    else:
      self.logger.error('Config set failed')
      exit(1)

  def connect(self):
    """
    Connects the client to RSockServer.
    Sends `"CONNECT"` to RSockServer.
    """
    self.logger.info('Starting the io_copy loop')
    self.rsock_socket.sendall(bytes('CONNECT', 'utf-8'))
    
    # start the io_copy loop
    self.rsock_thread_running = True
    self.rsock_thread = threading.Thread(target=self.io_copy, args=("server", ))
    self.rsock_thread.start()

    # start the io_copy loop
    self.client_thread_running = True
    self.client_thread = threading.Thread(target=self.io_copy, args=("client", ))
    self.client_thread.start()

  def io_copy(self, direction):
    """
    This is the main loop that handles the data transfer between client and server.
    """
    self.logger.info('Starting {} io_copy'.format(direction))

    if direction == 'client':
      client_socket = self.client_socket
      server_socket = self.rsock_socket

    elif direction == 'server':
      client_socket = self.rsock_socket
      server_socket = self.client_socket

    while True:
      try:
        data = client_socket.recv(1024)
        if data:
          # self.logger.info('{} data: {}'.format(direction, data))
          server_socket.sendall(data)
        else:
          self.logger.info('{} connection closed'.format(direction))
          break
      except Exception as e:
        self.logger.error('Error in {} io_copy: {}'.format(direction, e))
        break
    self.logger.info('Stopping {} io_copy'.format(direction))

def create_connection(local_port, instance_id, instance_port, logger: FileLogger, listen = 1):
  listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  listen_socket.bind(('localhost', local_port))
  listen_socket.listen(listen)

  connection_id = 0

  while True:
    logger.info('Waiting for client')
    client_socket, _ = listen_socket.accept()
    logger.info('Client connected')

    connection_id += 1
    logger.info(f'Total clients connected -> {connection_id}')
    # create the client
    pwd = secret.get("access_token")

    client = RSockClient(
      connection_id = connection_id,
      client_socket = client_socket,
      instance = instance_id,
      instance_port = instance_port,
      auth = pwd,
      logger = logger,
      secure = True
    )

    # start the client
    client.connect()

def port_in_use(port: int) -> bool:
  import socket
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    return s.connect_ex(('localhost', port)) == 0

def tunnel(ssh: int, *apps_to_ports: List[str], i: str):
  """the nbox way to SSH into your instance, by default ``"jupyter": 8888 and "mlflow": 5000``

  Usage:
    tunn.py 8000 notebook:8000 2000:8001 -i "nbox-dev"

  Args:
    ssh: Local port to connect SSH to
    *apps_to_ports: A tuple of values ``<app_name/instance_port>:<localport>``.
      For example, ``jupyter:8888`` or ``2001:8002``
    i(str): The instance to connect to
    pwd (str): password to connect to that instance.
  """

  import sys

  if sys.platform.startswith("linux"):  # could be "linux", "linux2", "linux3", ...
    pass
  elif sys.platform == "darwin":
    pass
  elif sys.platform == "win32":
    # Windows (either 32-bit or 64-bit)
    raise Exception("Windows is unsupported platform, raise issue: https://github.com/NimbleBoxAI/nbox/issues")
  else:
    raise Exception(f"Unkwown platform '{sys.platform}', raise issue: https://github.com/NimbleBoxAI/nbox/issues")

  folder = U.join(NBOX_HOME_DIR, "tunnel_logs")
  os.makedirs(folder, exist_ok=True)
  filepath = U.join(folder, f"tunnel_{i}.log")
  file_logger = FileLogger(filepath)

  
  # ===============

  default_ports = {
    "jupyter": 8888,
    "mlflow": 5000,
  }
  apps = {} # <localport-cloudport>
  for ap in apps_to_ports:
    app, port = ap.split(':')
    port = int(port)
    if not app or not port:
      raise ValueError(f"Invalid app:port pair {ap}")
    try:
      apps[int(app)] = port
    except:
      if app not in default_ports:
        raise ValueError(f"Unknown '{app}' should be either integer or one of {', ' .join(default_ports.keys())}")
      apps[port] = default_ports[app]
  apps[ssh] = 22 # hard code

  ports_used = []
  for k in apps:
    if port_in_use(k):
      ports_used.append(str(k))

  if ports_used:
    raise ValueError(f"Ports {', '.join(ports_used)} are already in use")

  # check if instance is the correct one
  instance = Instance(i)
  if not instance.state == "RUNNING":
    raise ValueError("Instance is not running")
  nbx_logger.info(f"password: {instance.open_data['ssh_pass']}")

  # create the connection
  threads = []
  for local_port, cloud_port in apps.items():
    nbx_logger.info(f"Creating connection from {cloud_port} -> {local_port}")
    t = threading.Thread(target=create_connection, args=(local_port, instance.instance_id, cloud_port, file_logger, 1))
    t.start()
    threads.append(t)

  try:
    # start the ssh connection on terminal
    import subprocess
    nbx_logger.info(f"Starting SSH ... for graceful exit press Ctrl+D then Ctrl+C")
    subprocess.call(f'ssh -p {ssh} ubuntu@localhost', shell=True)
  except KeyboardInterrupt:
    nbx_logger.info("KeyboardInterrupt, closing connections")
    for t in threads:
      t.join()

  sys.exit(0) # graceful exit
