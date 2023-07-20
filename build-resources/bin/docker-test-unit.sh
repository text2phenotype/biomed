#!/usr/bin/env bash

set -e
set -o pipefail

export PATH="$HOME/.local/bin:$PATH"

# Source the bash.utils
source "${APP_BASE_DIR}/bin/bash.utils"

log.info "> Starting 'docker-test-unit.sh' ($(date))"

### Python Sanity Check
log.info "> Using Python..."
which python
python --version

log.info "> Upgrading PIP..."
python -m pip install -U pip

log.info "> Using PIP:"
which pip

log.info "> Using PIP version:"
pip --version
### Python Sanity Check

# Skip slow tests
export SKIP_SLOW_TESTS_ON_DOCKER_BUILD=1

log.info "> Running unit tests..."
python setup.py test -a biomed/tests/unit
exit_code=$?

log.info "> Unit tests complete. Exited ('${exit_code}')"

log.info "> Finished 'docker-test-unit.sh' ($(date))"
exit ${exit_code}
