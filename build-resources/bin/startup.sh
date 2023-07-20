#!/usr/bin/env bash

set -e
set -o pipefail

# Source the bash.utils
source "${APP_BASE_DIR}/bin/bash.utils"

log.info "> Starting 'startup.sh' ($(date))"

export PATH="${APP_BASE_DIR}/.local/bin:$PATH"

# Port Nginx will listen on
export NGINX_LISTEN_PORT="${NGINX_LISTEN_PORT:-8080}"

# Downcase the these variables for consistent usage
APP_SERVICE=$( tr '[:upper:]' '[:lower:]' <<<"${APP_SERVICE}" )

### Python Sanity Check
log.info "> Using Python..."
which python
python --version

log.info "> Using PIP:"
which pip

log.info "> Using PIP version:"
pip --version
### Python Sanity Check

log.info "> Starting ${APP_NAME} as service ${APP_SERVICE}..."

trap "" SIGHUP SIGINT SIGTERM 30

case "${APP_SERVICE}" in
  worker-task-summary)
    time python "${APP_BASE_DIR}/biomed/workers/summary/start_worker.py"
  ;;

  worker-summary-bladder)
    time python "${APP_BASE_DIR}/biomed/workers/bladder_risk/start_summary_worker.py"
  ;;

  worker-task-deid)
    time python "${APP_BASE_DIR}/biomed/workers/deid/start_worker.py"
  ;;

  worker-model-bladder-risk)
    time python "${APP_BASE_DIR}/biomed/workers/bladder_risk/start_model_worker.py"
  ;;

  worker-model-demographic)
    time python "${APP_BASE_DIR}/biomed/workers/demographics/start_worker.py"
  ;;

  worker-model-doctype)
    time python "${APP_BASE_DIR}/biomed/workers/doc_type/start_worker.py"
  ;;

  worker-model-date-of-service)
    time python "${APP_BASE_DIR}/biomed/workers/date_of_service/start_worker.py"
  ;;

  worker-model-oncology)
    time python "${APP_BASE_DIR}/biomed/workers/oncology_only/start_worker.py"
  ;;

  worker-model-phi)
    time python "${APP_BASE_DIR}/biomed/workers/phi_token/start_worker.py"
  ;;

  worker-model-drug)
    time python "${APP_BASE_DIR}/biomed/workers/drug/start_worker.py"
  ;;

  worker-model-covid-lab)
    time python "${APP_BASE_DIR}/biomed/workers/covid_lab/start_worker.py"
  ;;

  worker-model-family-history)
    time python "${APP_BASE_DIR}/biomed/workers/family_history/start_worker.py"
  ;;

  worker-model-device-procedure)
    time python "${APP_BASE_DIR}/biomed/workers/device_procedure/start_worker.py"
  ;;

  worker-model-disease-sign)
    time python "${APP_BASE_DIR}/biomed/workers/disease_sign/start_worker.py"
  ;;

  worker-model-genetics)
    time python "${APP_BASE_DIR}/biomed/workers/genetics/start_worker.py"
  ;;

  worker-model-icd10-diagnosis)
    time python "${APP_BASE_DIR}/biomed/workers/icd10_diagnosis/start_worker.py"
  ;;

  worker-model-image-finding)
    time python "${APP_BASE_DIR}/biomed/workers/imaging_finding/start_worker.py"
  ;;

  worker-model-lab)
    time python "${APP_BASE_DIR}/biomed/workers/lab/start_worker.py"
  ;;

  worker-model-smoking)
    time python "${APP_BASE_DIR}/biomed/workers/smoking/start_worker.py"
  ;;

  worker-model-vital)
    time python "${APP_BASE_DIR}/biomed/workers/vital_sign/start_worker.py"
  ;;

  worker-model-sdoh)
    time python "${APP_BASE_DIR}/biomed/workers/sdoh/start_worker.py"
  ;;

  worker-train-test)
    time python "${APP_BASE_DIR}/biomed/workers/train_test/start_worker.py"
  ;;

  worker-task-reassembler)
    time python "${APP_BASE_DIR}/biomed/workers/reassembler/start_worker.py"
  ;;

  worker-task-pdf-embedder)
    time python "${APP_BASE_DIR}/biomed/workers/pdf_embedder/start_worker.py"
  ;;

  biomed-api|models-metadata-api)
    export GUNICORN_SOCKET_PATH="unix:${APP_BASE_DIR}/biomed.sock"
    export GUNICORN_WORKER_CLASS="${GUNICORN_WORKER_CLASS:-gthread}"
    # Default threads to 1 or it forces gthreads as the worker class
    export GUNICORN_WORKER_THREADS="${GUNICORN_WORKER_THREADS:-100}"
    export GUNICORN_WORKER_TIMEOUT="${GUNICORN_WORKER_TIMEOUT:-300}"
    export GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"

    declare nginx_default_conf="/etc/nginx/sites-enabled/default"

    log.info ">> Installing nginx proxy config..."
    log.info "GUNICORN_SOCKET_PATH: ${GUNICORN_SOCKET_PATH}"

    j2 "${APP_BASE_DIR}/build-resources/config/nginx-proxy.conf.j2" > "${nginx_default_conf}"
    log.info "> nginx configuration complete."

    log.info "> Starting nginx..."
    /usr/sbin/nginx
    log.info "> Launching guincorn workers..."
    log.info "Worker Class  : ${GUNICORN_WORKER_CLASS}"
    log.info "Worker Timeout: ${GUNICORN_WORKER_TIMEOUT}"
    log.info "Worker Number : ${GUNICORN_WORKERS}"
    if [[ "$GUNICORN_WORKER_CLASS" == "gthread" ]]; then
      log.info "Worker Threads : ${GUNICORN_WORKER_THREADS}"
    fi

    if [[ "$APP_SERVICE" == "models-metadata-api" ]]; then
      gunicorn_cmd="biomed.__main__:create_app(models_metadata_api=True)";
    else
      gunicorn_cmd="biomed.__main__:create_app()";
    fi

    time gunicorn \
      --timeout ${GUNICORN_WORKER_TIMEOUT} \
      --workers ${GUNICORN_WORKERS} \
      --worker-class ${GUNICORN_WORKER_CLASS} \
      --threads ${GUNICORN_WORKER_THREADS} \
      --bind ${GUNICORN_SOCKET_PATH} ${gunicorn_cmd};

  ;;

  *)
    log.error "> The service name '${APP_SERVICE}' was not recognized. Not starting a service!"
  ;;

esac

log.warn "> ${APP_NAME} as service ${APP_SERVICE} stopped!"

log.info "> Finished 'startup.sh' ($(date))"
