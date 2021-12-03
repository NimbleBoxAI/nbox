nbox Serving
============

This is the nbox-serving module, this is the open source version of what we have hosted
as our NBX Deploy product and is built on top of the FastAPI framework.

The computing paradigm that I am aiming for is called YoCo. YoCo is a new paradigm which
is built specially for post client-server era and resembles an RPC in practice. You can
read more `here <https://yashbonde.github.io/general-perceivers/remote.html>`_.

To start ``nbox`` as server, you can run the following command:

.. code-block:: bash

    NBOX_MODEL_PATH=filepath.nbox uvicorn server:app

More on this later.
