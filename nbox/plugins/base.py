from nbox.utils import logger, lo

def import_error(*packages) -> Exception:
  return ImportError(f"Please install the {', '.join(packages)} packages to use this nbox plugin")

def experimental_warning(name: str):
  logger.warning(f"Seem's like a curious mind is trying to use a new feature: {name}. This feature is experimental and can be unstable")
