# AI Box

A library that makes using a host of models provided by the opensource community a lot more easier. 

> The entire purpose of this package is to make inference chill.

# Changelog

- 01/09/2021: Adding pretrained models from `transformers` and add relevant tests in a new branch called `nlp`

## Things for Repo

- Using [`poetry`](https://python-poetry.org/) for proper package management as @cshubhamrao says.
  - Add new packages with `poetry add <name>`. Do not add `torch`, `tensorflow` and others, useless burden to manage those.
  - When pushing to pypi just do `poetry build && poetry publish` this manages all the things around
- Install `pytest` and then run `pytest tests/ -v`.
- Using `black` for formatting, VSCode to the moon.
