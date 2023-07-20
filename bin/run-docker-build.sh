#!/usr/bin/env bash

set -e
set -o pipefail

echo "> Starting 'run-docker-build.sh' ($(date))"

echo "> Working directory: $(pwd)"

if [[ -x build-tools/bin/suite-runner ]]; then
    # export DOCKER_FROM_REPO='base-image'
    export PATH="/$HOME/.local/bin:/app/.local/bin:/build-tools/bin:${PATH}"
    export DOCKER_NORMAL_BUILD='true'
    export DOCKER_BUILDKIT=1

    ### Python Sanity Check
    echo "> Using system Python3:"
    which python3
    python3 --version

    echo "> Upgrading PIP..."
    python3 -m pip install -U pip

    echo "> Using PIP:"
    which pip3

    echo "> These PIP versions should match!"
    pip --version
    pip3 --version
    ### Python Sanity Check

    # This allows a developer to demand that models be installed locally
    # into the container. This is useful in train1 where there are multiple
    # pipes in use at the same time, where SSS could cause issues.
    if [[ "${LOCAL_MODELS+x}" ]]; then
      echo "> \$LOCAL_MODELS was set, installing models locally with DVC..."

      echo "> Installing DVC"
      pip install dvc[s3]

      echo "> DVC Pull"
      time dvc pull -f -j 9 -v

      echo "> Removing DVC"
      pip uninstall dvc -y
    fi

    echo "> Starting suite-runner"
    build-tools/bin/suite-runner

else
    echo "The script does not exist or did not have execute permissions: build-tools/bin/suite-runner"
    stat build-tools/bin/suite-runner || true
fi

echo "> Finished 'run-docker-build.sh' ($(date))"
