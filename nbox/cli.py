from .auth import secret
import webbrowser

def open_home():
  """Open current NBX platform"""
  webbrowser.open(secret.get("nbx_url"))

