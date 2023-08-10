#!/usr/bin/env bash
if [[ -z $DIR ]]; then
    DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
fi
ENV_FILE=$DIR/conda-env.yml

required_version () {
    name=$1

    awk -F'[ ="]*' "/$name/"'{print $4; exit}' $ENV_FILE
}

current_version () {
    name=$1
    get_current_awk_script=$2

    if [[ -z $get_current_awk_script ]]; then
        get_current_awk_script='NR==1{print $2}'
    fi
    $name --version | awk "$get_current_awk_script"
}

check_version () {
    name=$1
    get_current_awk_script=$2

    REQUIRED=`required_version "$name"`
    CURRENT=`current_version "$name" "$get_current_awk_script"`

    if [[ $CURRENT != "$REQUIRED"* ]]; then
        >&2 echo "$name version mismatch: current \"$CURRENT\", required \"$REQUIRED\""
        return 1
    fi
}

check_versions () {(
    set -e

    check_version python
    check_version black
    check_version flake8 'NR==1{print $1}'
    check_version mypy
    check_version pytest
)}

run_static_checks () {(
    set -ex

    black . --check
    flake8 . --count --max-line-length=127 --show-source --statistics --ignore=E741,W503,E203
    mypy optson/ docs/jupytext_notebooks/
)}

run_pytest () {(
    set -ex

    py.test --cov=optson/ tests/
)}
