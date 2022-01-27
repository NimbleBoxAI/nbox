# Documentation

Writing docs is hard, and a tedious job, but one well worth for refining communication of thought! Here are some fellow tools to help you do that:

1. (m2r)[https://github.com/miyakogi/m2r]: Writing `rst` is hard, convert your markdown to rst!
2. You don't need anything else love.

Have fun reading the hidden stories!

Use this command in `/nbox`: `sphinx-apidoc -o docs/source/ ./nbox/ -M -e` to add base plates for something new. Use https://docutils.sourceforge.io/docs/user/rst/quickref.html to get quick tools.

<<<<<<< Updated upstream
<!-- https://pdoc3.github.io/pdoc/ -->
=======
<!-- https://pdoc3.github.io/pdoc/ -->

Wanted to try building watcher in Go. So start like this:

```
[Tab 1] $ go run .
[Tab 2] $ ./watcher.py
[Tab 3] $ cd build/html && python3 -m http.server 80
```

Currently watches `docs/` and `nbox/` directories.

If you want to export the go code you can cross-compile it for other architectures.
>>>>>>> Stashed changes
