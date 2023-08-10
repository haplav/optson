#!/usr/bin/env bash
set -ex
for tag in ci ci-jax ci-torch ci-salvus-deps; do
  docker build --file Dockerfile-$tag --tag optson:$tag .
done
