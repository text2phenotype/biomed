#!/usr/bin/env bash

set -e
set -o pipefail

###
# Import utilities
export PATH="/$HOME/.local/bin:./build-tools/bin:${PATH}"
source "./build-tools/bin/build.utils"

log.info "> Starting 'run-tests-integration.sh' ($(date))"

###
# Variables

# UNIVERSE_IS_VERBOSE enables log level INFO.
UNIVERSE_IS_VERBOSE=true

log.info "> Working directory: $(pwd)"

### Python Sanity Check
log.info "> Using Python3..."
which python3
python3 --version

log.info "> Upgrading PIP..."
python3 -m pip install -U pip

log.info "> Using PIP:"
which pip
### Python Sanity Check

log.info "> Install docker-compose..."
pip install docker-compose

TMP_PROJ="$(date +%s)-$RANDOM"

log.info "> Starting integration testing..."

log.info "> Start integration testing..."

# Build any required resources
log.info "> docker-compose build..."
docker-compose -p $TMP_PROJ --file docker-compose-tests-integration.yaml build

# Stand up the test stack
log.info "> docker-compose up..."
docker-compose -p $TMP_PROJ --file docker-compose-tests-integration.yaml up --exit-code-from biomed
integration_test_result=$?

# Cleanup docker-compose
log.info "> docker-compose down..."
docker-compose -p $TMP_PROJ --file docker-compose-tests-integration.yaml down

log.info "> Integration testing complete. ('${integration_test_result}')"

log.info "> Finished 'run-tests-integration.sh' ($(date))"
exit ${integration_test_result}
