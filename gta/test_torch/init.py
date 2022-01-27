import sys
from nbox.utils import folder, join, _isthere

sys.path.append(join(folder(folder(__file__))))

from common import ModuleNotFound

def main():
  if not _isthere('torch'):
    ModuleNotFound('torch')
