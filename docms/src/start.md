# nbox {% gradientText text="Documentation"/%} {% .marginb8 %}

> Writer is for whom writing is more difficult than others. - Thomas Mann

Welcome to the documentation of `nbox` which is our "code frontend" (not just another client library). What makes `nbox` special is that it not only contains code for gRPC/REST clients but also application layer modules which you can use to build your own Apps. This is unlike anything you would have used before.

Here's a very quick primer on NimbleBox:

{% Table headings=["Layer","Modules","Notes","Where"] spacing=[2,2,6,1] values=[["Application","`Operator`","Use this to build any arbitrary workflow, deploy and manage through CLI or programatically","`nbox`"],["Client","`Lmao`,`RelicsNBX`,`Job`,`Serve`", "Clients that connect to our different backends","`nbox`"],["Webserver (L1 Backend)", "`nbox_ws_v1`", "Connect using REST endpoints, manages RBAC","NimbleBox"], ["Services (L2 Backends)", "`nbox_grpc_stub`", "All the seperate services you can connect to in isolation","S3, ECR, ..."]] /%}

This is important to note because `nbox` is intented to be used where it fits in your workflow and not the other way around. Sometimes all it needs to do is be a single import in the corner of your codebase, or the first module in your workflow. Whatever it is, `nbox` is not going to intrude where is shouldn't. So what are all the things you need to be aware of when you are using this:

- `Operator` is a powerful class that can be used to describe any kind of workflow that you want
- You can talk to big VMs using `Instance` class, which can also be used to SSH through CLI
- `Job` can be used to work with NBX-Jobs service which is essentially a batch processing + serverless endpoint
- `Serve` can be used to talk to NBX-Deploy service for live API endpoints
- All the storage can be accessed via `RelicsNBX`, to connect your own bucket reach out via Intercom
- `LMAO` is our experimental Monitoring and Alerting service, it is currently available in private beta

You can get an idea how by just playing around these building blocks you can create any kind of MLOps pipeline that you want to.
