import json
import sys
import os
from typing import List
from time import sleep

from .network import deploy_model
from .auth import init_secret, secret
from .utils import get_random_name, NBOX_HOME_DIR, join
from .jobs import get_instance, Instance

def status(loc = None):
    from .jobs import print_status
    print_status(f"https://{'' if not loc else loc+'.'}nimblebox.ai")


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

    import socket
    import logging

    import ssl
    import threading
    import certifi

    log_ = open(join(NBOX_HOME_DIR, "tunnel.log"), "w")
    logger = lambda x: log_.write(x + "\n")

    class RSockClient:
        """
        This is a RSockClient. It handels the client socket where client is the user application trying to connect to "client_port"
        Connects to RSockServer listening on localhost:886.
        RSockServer recieves instructions as a string and sends back a response.
        RSockServer requires following steps to setup
        First,
            Authentication:
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

        def __init__(self, connection_id, client_socket, instance, instance_port, auth, secure=False):
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

            self.client_auth = False
            self.rsock_thread_running = False
            self.client_thread_running = False


            self.log('Starting client')
            self.connect_to_rsock_server()
            self.log('Connected to RSockServer')
            self.authenticate()
            self.log('Authenticated client')
            self.set_config()
            self.log('Client init complete')
        
        def log(self, message, level=logging.INFO):
            logger(f"[Client_ID: {self.connection_id}] {message}")
            # print(f"[Client_ID: {self.connection_id}] {message}")
        
        def connect_to_rsock_server(self):
            """
            Connects to RSockServer.
            """
            self.log('Connecting to RSockServer', logging.DEBUG)
            rsock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            rsock_socket.connect(('rsocks.nimblebox.ai', 886))

            if self.secure:
                self.log('Starting SSL')
                self.rsock_socket = ssl.wrap_socket(rsock_socket, ca_certs=certifi.where(), cert_reqs=ssl.CERT_REQUIRED)
            else:
                self.rsock_socket = rsock_socket

        def authenticate(self):
            """
            Authenticates the client.
            Sends `"AUTH~{AUTH_TOKEN}"` to RSockServer.
            """
            self.log('Authenticating client')
            self.rsock_socket.sendall(bytes('AUTH~{}'.format(self.auth), 'utf-8'))
            auth = self.rsock_socket.recv(1024)
            auth = auth.decode('utf-8')
            if auth == 'OK':
                self.log('Client authenticated')
            else:
                self.log('Client authentication failed', logging.ERROR)
                self.client_auth = False
                exit(1)
        
        def set_config(self):
            """
            Sets the config of the client.
            Sends `"SET_CONFIG~{instance}~{instance_port}"` to RSockServer.
            """
            self.log('Setting config')
            self.rsock_socket.sendall(bytes(f'SET_CLIENT~{self.instance}~{self.instance_port}', 'utf-8'))
            config = self.rsock_socket.recv(1024)
            config = config.decode('utf-8')
            self.log('Config set to {}'.format(config))
            if config == 'OK':
                self.log('Config set')
            else:
                self.log('Config set failed', logging.ERROR)
                exit(1)

        def connect(self):
            """
            Connects the client to RSockServer.
            Sends `"CONNECT"` to RSockServer.
            """
            self.log('Starting the io_copy loop')
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
            self.log('Starting {} io_copy'.format(direction))

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
                        # self.log('{} data: {}'.format(direction, data))
                        server_socket.sendall(data)
                    else:
                        self.log('{} connection closed'.format(direction))
                        break
                except Exception as e:
                    self.log('Error in {} io_copy: {}'.format(direction, e), logging.ERROR)
                    break
            self.log('Stopping {} io_copy'.format(direction))


    def create_connection(local_port, instance_id, instance_port, listen = 1):
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.bind(('localhost', local_port))
        listen_socket.listen(listen)

        connection_id = 0

        while True:
            logger('Waiting for client')
            client_socket, _ = listen_socket.accept()
            logger('Client connected')

            connection_id += 1
            logger('Total clients connected -> '.format(connection_id))
            # create the client
            pwd = secret.get("access_token")

            client = RSockClient(connection_id, client_socket, instance_id, instance_port, pwd, True)

            # start the client
            client.connect()

    def port_in_use(port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
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
    instance = Instance(i, loc = "test-3")
    if not instance.state == "RUNNING":
        raise ValueError("Instance is not running")
    passwd = instance.open_data["ssh_pass"]
    logging.info(f"password: {passwd}")

    # create the connection
    threads = []
    for local_port, cloud_port in apps.items():
        logging.info(f"Creating connection from {cloud_port} -> {local_port}")
        t = threading.Thread(target=create_connection, args=(local_port, instance.instance_id, cloud_port, 1))
        t.start()
        threads.append(t)

    try:
        # start the ssh connection on terminal
        import subprocess
        logging.info(f"Starting SSH ... for graceful exit press Ctrl+D then Ctrl+C")
        subprocess.call(f'ssh -p {ssh} ubuntu@localhost', shell=True)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt, closing connections")
        for t in threads:
            t.join()

    # TODO:@yashbonde Make Platform agnostic
    subprocess.run(["ssh-keygen", "-R", f"localhost[{ssh}]"])
    sys.exit(0) # graceful exit


def deploy(
    config_path: str = None,
    model_path: str = None,
    model_name: str = None,
    nbox_meta: str = None,
    deployment_type: str = None,
    deployment_id: str = None,
    deployment_name: str = None,
    convert_args: str = None,
    wait_for_deployment: bool = False,
    print_in_logs: bool = False,
    username: str = None,
    password: str = None,
    nbx_home_url="https://nimblebox.ai",
):
    r"""Deploy a model from nbox CLI. Add this to your actions and see the magic happen!
    If you are using a config file then data will be loaded from there and other kwargs will be ignored.

    Args:
        config_path (str, optional): path to your config file, if provided, everything else is ignored
        model_path (str, optional): path to your model
        model_name (str, optional): name of your model
        nbox_meta ([str, dict], optional): path to your nbox_meta json file, if None tries to find by replacing
          ``model_path`` extension with ``.json`` or can be a dict if using ``config_path``
        deployment_type (str, optional): type of deployment, can be one of: ovms2, nbox
        convert_args (str, optional): if using ovms2 deployment type, you must pass convertion CLI args
        wait_for_deployment (bool, optional): wait for deployment to finish, if False this behaves async
        print_in_logs (bool, optional): print logs in stdout
        username (str, optional): your NimbleBox.ai username
        password (str, optional): your password for corresponding ``username``

    Usage:
        Convert each ``kwarg`` to ``--kwarg`` for CLI. eg. if you want to pass value for ``model_path`` \
            in cli it becomes like ``... --model_path="my_model_path" ...``

        .. code-block:: bash

            python3 -m nbox deploy --model_path="path/some/sklearn.pkl" \
                --nbox_meta="path/to/nbox_meta.json"

    Raises:
        ValueError: if ``deployment_type`` is not supported
        AssertionError: if model path is not found or ``nbox_meta`` is incorrect
        Exception: if ``deployment_type == "ovms2"`` but ``convert_args`` is not provided
    """
    from nbox.auth import secret  # it can refresh so add it in the method

    if secret is None or secret.get("access_token", None) == None:
        # if secrets file is not found
        assert username != None and password != None, "secrets.json not found need to provide username password for auth"
        access_token = get_access_token(nbx_home_url, username, password)
        create_secret_file(username, access_token, nbx_home_url)
        init_secret()  # reintialize secret variable as it will be used everywhere

    if config_path != None:
        with open(config_path, "r") as f:
            config = json.load(f)
        config.pop("config_path", None) # remove recursion
        deploy(**config)

    else:
        # is model path valid and given
        if not os.path.exists(model_path):
            assert os.path.exists(model_path), "model path not found"

        # check if nbox_meta is correct
        if nbox_meta == None:
            nbox_meta = ".".join(model_path.split(".")[:-1]) + ".json"
            print("Trying to find nbox meta at path:", nbox_meta)
            assert os.path.exists(nbox_meta), "nbox_meta not provided"
        else:
            raise ValueError("nbox_meta is not supported yet")

        if isinstance(nbox_meta, str):
            if not os.path.exists(nbox_meta):
                raise ValueError(f"Nbox meta path {nbox_meta} does not exist. see nbox.Model.get_nbox_meta()")
            with open(nbox_meta, "r") as f:
                nbox_meta = json.load(f)
        else:
            assert isinstance(nbox_meta, dict), "nbox_meta must be a dict"

        # validation of deployment_type
        assert deployment_type in ("ovms2", "nbox"), "Deployment type must be one of: ovms2, nbox"
        if deployment_type == "ovms2":
            assert convert_args is not None, (
                "Please provide convert args when using OVMS deployment, "
                "use nbox.Model.deploy(deployment_type == 'ovms2') if you are unsure!"
            )

        # one click deploy
        model_name = get_random_name().replace("-", "_") if model_name == None else model_name
        endpoint, key = deploy_model(
            model_path, model_name, deployment_type, nbox_meta, wait_for_deployment, convert_args, deployment_id, deployment_name
        )

        # print to logs if needed
        if wait_for_deployment and print_in_logs:
            print(" Endpoint:", endpoint)
            print("Model Key:", key)
