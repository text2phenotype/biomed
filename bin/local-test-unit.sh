#!/usr/bin/env bash

set -e
set -o pipefail

echo "> Starting 'local-test-unit.sh' ($(date))"

if [[ -n $SKIP_TESTS || -n $SKIP_UNIT_TESTS ]]; then
  echo "> Not running unit tests..."
  exit 0
fi

# ENV
export BIOMED_PRELOAD=False
export MDL_BIOM_PRELOAD=False
export MDL_BIOM_SMOKING_CLINICAL=True
export MDL_BIOM_USE_STORAGE_SVC=True
export MDL_BIOM_VITAL_SIGNS_CLINICAL=True

cd $WORKSPACE
echo "> Working directory: $(pwd)"

echo "> Installing DVC globally"
pip install dvc[s3]

echo "> Starting DVC pull in the background"
dvc pull -f -j9 &
dvc_biomed=$!

### Python Sanity Check
echo "> Using Python3..."
which python3
python3 --version

VENV_SRC_DIR=${VENV_SRC_DIR:=/venv}

echo "> Creating Python3 virtualenv with that version"
python3 -m virtualenv $VENV_SRC_DIR
source $VENV_SRC_DIR/bin/activate

echo "> Using virtualenv Python:"
which python

echo "> These Python versions should match!"
python --version
$VENV_SRC_DIR/bin/python --version

echo "> Upgrading virtualenv PIP..."
python -m pip install -U pip

echo "> Using virtualenv PIP:"
which pip

echo "> These PIP versions should match!"
pip --version
$VENV_SRC_DIR/bin/pip --version
### Python Sanity Check

echo "> Installing text2phenotype-py"
time pip install -e ../text2phenotype-py

echo "> Installing Biomed requirements explicitly"
time pip install -r requirements-test.txt

echo "> Installing Biomed"
time pip install -e .

echo "> $(date)"
echo "> Waiting for DVC pull to finish in the background"
wait ${dvc_biomed}
echo "> $(date)"

echo "> Starting Biomed unit tests..."

# Split unit tests to several python-processes to reduce memory consumption

pip install pytest-custom_exit_code

for test_dir in $(ls biomed/tests/unit/*/ -d); do
  echo "> Run unit tests from \"${test_dir}\" directory"

  time python -m pytest "${test_dir}" \
    --junitxml="${UNIT_TESTS_REPORT_FILE:-junit-report-unit.xml}" \
    --disable-warnings \
    --suppress-no-test-exit-code
done

echo "> Finished 'local-test-unit.sh' ($(date))"
