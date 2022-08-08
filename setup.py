import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nbox",
    version="0.9.17",
    author="NimbleBox.ai",
    author_email="research@nimblebox.ai",
    description="Make inference chill again!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NimbleBoxAI/nbox",
    project_urls={
        "Bug Tracker": "https://github.com/NimbleBoxAI/nbox/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ApacheV2",
        "Operating System :: OS Independent",
    ],
    package_dir={"nbox": "nbox"},
    python_requires=">=3.6",
)
