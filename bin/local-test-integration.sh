#!/usr/bin/env bash

set -e
set -o pipefail

echo "> Starting 'local-test-integration.sh' ($(date))"

if [[ -n $SKIP_TESTS || -n $SKIP_INT_TESTS ]]; then
  echo "> Not running integration tests..."
  exit 0
fi

# ENV
export BIOMED_PRELOAD=False
export MDL_BIOM_DATA_ROOT="${WORKSPACE}/S3/biomed-data"
export MDL_BIOM_PRELOAD=False
export MDL_BIOM_SMOKING_CLINICAL=True
export MDL_BIOM_USE_STORAGE_SVC=True
export MDL_BIOM_VITAL_SIGNS_CLINICAL=True
export MDL_COMN_BIOMED_API_BASE=http://0.0.0.0:8080
export MDL_COMN_DATA_ROOT="${MDL_BIOM_DATA_ROOT}"
export MDL_COMN_STORAGE_CONTAINER_NAME=biomed-data
export MDL_COMN_USE_STORAGE_SVC=True
export MDL_FEAT_API_BASE=http://0.0.0.0:8081
export MDL_FEAT_UMLS_REQUEST_MODE=false

echo "> Downloading S3 data from 's3://${MDL_COMN_STORAGE_CONTAINER_NAME}' to '${MDL_COMN_DATA_ROOT}/'"
s3_files=( "Customer" "pubmed" "emr" "himss" )
pids=()
for s3_file in ${s3_files[@]}; do
  aws s3 sync --only-show-errors --quiet s3://${MDL_COMN_STORAGE_CONTAINER_NAME}/${s3_file} ${MDL_BIOM_DATA_ROOT}/${s3_file} &
  pids+=($!)
done

cd ${WORKSPACE}
echo "> Working directory: $(pwd)"

echo "> Installing DVC globally"
pip install dvc[s3]

echo "> Pulling Biomed DVC data"
dvc pull -f -j9 &
pids+=($!)

echo "> Pulling FS DVC data"
cd ${WORKSPACE}/../feature-service
time dvc pull -f -j9 &
pids+=($!)
cd ${WORKSPACE}

### Python Sanity Check
echo "> Using Python3..."
which python3
python3 --version
which pip3
pip3 --version

VENV_SRC_DIR=${VENV_SRC_DIR:=/venv}

echo "> Creating Python 3 virtualenv"
python3 -m virtualenv $VENV_SRC_DIR

echo "> Activating virtualenv"
source $VENV_SRC_DIR/bin/activate

echo "> These versions should match:"
python3 --version
$VENV_SRC_DIR/bin/python --version
python --version

echo "> Using Python..."
which python

echo "> Upgrading virtualenv PIP..."
python -m pip install -U pip
### Python Sanity Check

echo "> Installing text2phenotype-py"
time pip install -e ../text2phenotype-py 

echo "> $(date)"
echo "> Waiting for background processes to finish"
for pid in ${pids[@]}; do
  wait ${pid}
done
echo "> $(date)"

echo ">>>>>>>>>>>>> Biomed"
echo "> Installing Biomed requirements explicitly"
time pip install -r requirements-test.txt

echo "> Installing Biomed"
time pip install -e .

echo "> Starting Model Metadata Service"
python biomed --models-metadata-service &> ${WORKSPACE}/biomed-stdout.log &

echo ">>>>>>>>>>>>> Feature Service"
cd ${WORKSPACE}/../feature-service
echo "> Working directory: $(pwd)"

echo "> Installing FS requirements explicitly"
time pip install -r requirements.txt

echo "> Installing Feature Service"
time pip install -e .

echo "> Installing NLTK"
python nltk_download.py

echo "> Setting MDL_FEAT_MODELS_ROOT to ${WORKSPACE}/feature-service/feature_service"
export MDL_FEAT_MODELS_ROOT="${WORKSPACE}/feature-service/feature_service"

echo "> Starting Feature Service"
python feature_service &> ${WORKSPACE}/feature_service-stdout.log &

echo "> Waiting 20 seconds for services to start..."
sleep 20

# Run tests
echo ">>>>>>>>>>>>> Biomed"
cd ${WORKSPACE}
echo "> Working directory: $(pwd)"

echo "> Starting Integration Tests"

# Split integration tests to several python-processes to reduce memory consumption

pip install pytest-custom_exit_code

for test_dir in $(ls biomed/tests/integration/*/ -d); do
  echo "> Run integration tests from \"${test_dir}\" directory"

  time python -m pytest "${test_dir}" \
    --junitxml="${INTEGRATION_TESTS_REPORT_FILE:-junit-report-integration.xml}" \
    --disable-warnings \
    --suppress-no-test-exit-code
done

# time python -m pytest biomed/tests/integration/ --junitxml="${INTEGRATION_TESTS_REPORT_FILE:-junit-report-integration.xml}"

echo "> Finished 'local-test-integration.sh' ($(date))"
