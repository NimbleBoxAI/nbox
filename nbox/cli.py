from .auth import secret
import webbrowser

def open_home():
  webbrowser.open(secret.get("nbx_url"))

