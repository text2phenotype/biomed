#!/usr/bin/env bash

### THIS FILE IS NOT USED

set -e
set -o pipefail

# Source the bash.utils
source "${APP_BASE_DIR}/bin/bash.utils"

log.info "> Starting 'docker-build.sh' ($(date))"

APPUSER=$(id -un 5001)

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

log.info "> Installing ${APP_BASE_DIR}/requirements.txt ..."
time pip install --no-cache-dir -r ${APP_BASE_DIR}/requirements.txt

log.info "> Installing ${APP_NAME}..."
time pip install -e ${APP_BASE_DIR}

log.info "> Setting up nginx config..."
cp -v "${APP_BASE_DIR}/build-resources/config/nginx.conf" "/etc/nginx/nginx.conf"
log.info "> Done Setting up nginx config"

log.info "> Removing default nginx site..."
rm -f "/etc/nginx/sites-available/default"
rm -f "/etc/nginx/sites-enabled/default"
log.info "> Done removing default nginx site"

log.info "> Chown'ing necessary files to $APPUSER ..."
touch /etc/nginx/sites-enabled/default
chown -R $APPUSER:$APPUSER "${APP_BASE_DIR}" /etc/nginx/sites-enabled/default /var/log/nginx /var/lib/nginx
log.info "> Done chown'ing necessary files to $APPUSER ..."

log.info "> ${APP_NAME} installation complete."

log.info "> Finished 'docker-build.sh' ($(date))"
