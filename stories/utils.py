import os

def hr(msg: str = "", symbol: str = "="):
    width = os.get_terminal_size().columns
    if len(msg) > width - 5:
        print(symbol * width)
        print(msg)
        print(symbol * width // 2)  # type: ignore
    else:
        print(symbol * (width - len(msg) - 1) + " " + msg)
