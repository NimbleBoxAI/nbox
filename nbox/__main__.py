import fire
from .cli import login, deploy

if __name__ == "__main__":
    fire.Fire({"login": login, "deploy": deploy})
