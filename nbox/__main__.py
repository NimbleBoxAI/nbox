import fire
from .cli import deploy

if __name__ == "__main__":
    fire.Fire({"deploy": deploy})
