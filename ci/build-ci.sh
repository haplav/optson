#!/usr/bin/env bash
set -ex
for tag in ci ci-jax-pytorch ci-salvus-deps; do
  docker build --file Dockerfile-$tag --tag optson:$tag .
done
