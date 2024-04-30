#!/usr/bin/env bash
set -ex
SERVER=registry.gitlab.com
REG=$SERVER/swp_ethz/optson

docker login $SERVER
for tag in ci ci-jax ci-torch ci-salvus-deps; do
  docker tag optson:$tag $REG:$tag
  docker push $REG:$tag
done
