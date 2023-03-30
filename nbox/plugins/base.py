def import_error(*packages) -> Exception:
  return ImportError(f"Please install the {', '.join(packages)} packages to use this nbox plugin")