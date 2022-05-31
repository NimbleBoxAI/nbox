QuickStart with NBX-Deploy
==========================

Deployment is a major bottleneck for DS/ML engineers to get their systems into production. Unless any
model is in production there really is no way to complete a project. ``nbox`` as an SDK makes deployment just one
command, **literally**\ ! 

Open your favorite IDE, whether ``jupyter`` or ``VSCode``. 

First you need to do some imports and you can load your own model, for this tutorial we want to deploy ``torchvision/resnet18``. Loading models is super easy, either use a publicly available models or bring in your own models.

.. code-block:: python

   import nbox
   from nbox.utils import get_image

   # load pretrained model
   model = nbox.load("torchvision/resnet18", pretrained=True,)

   # literally pass model a URL and it will process it
   image_url = "https://github.com/NimbleBoxAI/nbox/raw/master/tests/assets/cat.jpg"
   out = model(image_url)
   print(out[0].topk(5))

Now comes the cool part, i.e. deploying models. There is already support for deploying a tonne of models directly from nbox. For deployment you need to give it any input_object that can be used to perform trace (\ ``torchscript`` / ``ONNX``\ ). If you do not resize the input image the deployed model will have default shape, so for now we will resize the image to (244, 244) and then pass it for deployment.

.. code-block:: python

   image = get_image(image_url)      # get the PIL.Image object
   image = image.resize((244, 244))  # you can skip the following shape if the shape is already correct
   url, key = model.deploy(
       input_object=image,         # simply provide the input_object and watch the Terminal
       wait_till_deployment=True,  # this will return the url endpoint and key
   )

Now wait for the deployment to complete. You can check out the dashboard till then and once deployed you will see
the URL and API key for your model. Read more about deployment and runtimes `here <nbox.model.html>`_

.. code-block:: python

   print("url:", url)
   print("key:", key)

   # load the model and use it without any difference in API
   model = nbox.load(url, key)
   out = model(image_url)
   print(out[0].topk(5))

Use the model without being concerned with the API hits, as ``nbox`` handles it internally. So, you can see how easy it is to load a model, test it and deploy it in minutes. You can head over to technical documentation for further reading on this.
