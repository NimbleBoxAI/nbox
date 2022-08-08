Jobs + Deploy FAQs
==================

For FAQs regarding the platform click `here <https://docs.v2.nimblebox.ai/developer-tools/jobs/get-started>`_.

**How can I change resources?**

Open ``nbx_user.py`` file and you will see ``get_resource`` function that can used to modify your Pod requirements. Your Job
starts with reasonable default for most of the workloads.

**How do I transfer file to an object store like S3?**

We are working on ``nbox.Relic`` that will act as a single panel for multiple object stores like AWS S3, GCP Bukets, etc.

**How to run jobs at schedule?**

Open ``nbx_user.py`` file and you will see ``get_schedule`` function that can used to modify schedule for your Job. Note that
this is not used when uploading a serving.

