# nbx {% gradientText text="CLI"/%} {% .marginb8 %}

We have put special care and effort in ensuring that our CLI is as powerful as the importable modules and REST endpoints.

{% CallOut variant="tip" label="Use CLI **only** when you want to build pipelines using CI/CD tools. If you can define your workflows programatically, it's usually a better choice." /%}

We use [fire](https://github.com/google/python-fire) for building our CLI and it sometimes it means loading complete python modules and making a bunch of API calls which would make it a tad bit slower. Thanks to `fire` we have a 100% CLI-module-API parity! 
