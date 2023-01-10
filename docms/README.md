# Project New Documentation (Marquez)

All documentation engines suck, there is no consideration for the humans. So we took action.

This is what the folder structure looks like:

```
src/
  index.md
  changelog.md
  auto/
    nbox.cli.md
    nbox.lib.shell.md
    nbox.nbxlib.resource_contants.md
```

Will create a `gen/` folder with the exact same structure. All the module code is in `autogen/` folder which provides this simple convinience being tucked away in the folder.

This uses the `nbox.nbxlib.astea` module which is a simple human like AST parsing and exploration engine, so using our own code to solve our own problems. Another tricky piece is the src markdown file parsing.

## Software Requirements

- User should be able to specify exactly what piece of object they want.
- User should not need to specify anything as well, some things should just work.
- Have a dead simple build system, predictable, functional in flow, nothing fancy.
- Script should only run from repo root, no where else, remove the burden of path management.
- It should throw out relevant warnings as and when needed.
- It should not delete any file, unless told so.

## How to make nbox docs?

- **Only to create blanks**: Use this command only to create blank entries in the src/code folder note that this will remove eny existing piece of thing written:
  ```bash
  # we are going to ignore hyperloop and sublime modules since they mostly contain protobufs
  python3 build.py create_blanks --fresh --ignore_pat '["hyperloop", "sublime._yql", "sublime.proto"]'
  ```

- 

