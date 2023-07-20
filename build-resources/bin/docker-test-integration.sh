#!/usr/bin/env bash

set -e
set -o pipefail

# Source the bash.utils
source "${APP_BASE_DIR}/bin/bash.utils"

log.info "> Starting 'docker-test-integration.sh' ($(date))"

### Python Sanity Check
log.info "> Using Python..."
which python
python --version

log.info "> Upgrading PIP..."
python -m pip install -U pip

log.info "> Using PIP:"
which pip

log.info "> Using PIP version..."
pip --version
### Python Sanity Check

log.info "> Running integration tests..."
python -m pytest biomed/tests/integration/
exit_code=$?

log.info "> Integration tests complete. Exited ('${exit_code}')"

log.info "> Finished 'docker-test-integration.sh' ($(date))"
exit ${exit_code}
