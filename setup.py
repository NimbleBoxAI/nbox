import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aibox",
    version="0.0.1",
    author="NimbleBox.ai",
    author_email="admin@nimblebox.ai",
    description="Use community models more easily",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NimbleBoxAI/box_of_ai_tools/tree/python-lib",
    project_urls={
        "Bug Tracker": "https://github.com/NimbleBoxAI/box_of_ai_tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
