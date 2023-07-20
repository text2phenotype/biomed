#!/usr/bin/env bash

set -e
set -o pipefail

###
# Import utilities
export PATH="/$HOME/.local/bin:./build-tools/bin:${PATH}"
source "./build-tools/bin/build.utils"

log.info "> Starting 'run-tests-unit.sh' ($(date))"

###
# Variables

# UNIVERSE_IS_VERBOSE enables log level INFO.
UNIVERSE_IS_VERBOSE=true

log.info "> Working directory: $(pwd)"

log.info "> Start unit testing..."

### Python Sanity Check
log.info "> Using Python3..."
which python3
python3 --version

log.info "> Upgrading PIP..."
python3 -m pip install -U pip

log.info "> Using PIP:"
which pip

log.info "> Using PIP version..."
pip --version
### Python Sanity Check

log.info "> PIP list:"
pip list

TMP_PROJ="$(date +%s)-$RANDOM"

# Build any required resources
log.info "> docker-compose build..."
docker-compose -p $TMP_PROJ --file docker-compose-tests-unit.yaml build

# Stand up the test stack
log.info "> docker-compose up..."
docker-compose -p $TMP_PROJ --file docker-compose-tests-unit.yaml up --exit-code-from biomed
unit_test_result=$?

# Cleanup docker-compose
log.info "> docker-compose down..."
docker-compose -p $TMP_PROJ --file docker-compose-tests-unit.yaml down

log.info "> Unit testing complete. ('${unit_test_result}')"

log.info "> Finished 'run-tests-unit.sh' ($(date))"
exit ${unit_test_result}
