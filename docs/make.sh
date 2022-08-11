#!/bin/bash
export NBOX_NO_AUTH=1
export NBOX_NO_LOAD_GRPC=1
export NBOX_NO_LOAD_WS=1
make clean
make html

echo "Copying the new docs to the nbox-pages folder"
cp -r build/html/* ../../nbox-pages/