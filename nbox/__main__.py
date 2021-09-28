import fire

from nbox.cli import deploy, login

if __name__ == "__main__":
    fire.Fire({"login":login,"deploy":deploy})
