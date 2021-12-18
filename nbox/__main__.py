import fire
from .cli import deploy
from .jobs import jobs_cli, print_status

if __name__ == "__main__":
    fire.Fire({
        "deploy": deploy,
        "jobs": jobs_cli,
        "status": lambda loc = None: print_status(f"https://{'' if not loc else loc+'.'}nimblebox.ai")
    })
