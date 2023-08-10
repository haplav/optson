#!/usr/bin/env bash
set -e
DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $DIR/..
source $DIR/test-commons.sh
check_versions
pip install -q -e .
run_static_checks
run_pytest
