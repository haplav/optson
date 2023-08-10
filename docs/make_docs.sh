#!/usr/bin/env sh
make clean
rm -rf api/
sphinx-apidoc -o api ../optson -f -H "API Reference" -e
make html
