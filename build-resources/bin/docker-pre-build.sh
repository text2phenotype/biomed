#!/usr/bin/env bash

set -e
set -o pipefail

# Source the bash.utils
source "${APP_BASE_DIR}/bin/bash.utils"

log.info "> Starting 'docker-pre-build.sh' ($(date))"

log.info "> Starting ${APP_NAME} build..."

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

# log.info "> Installing ${APP_BASE_DIR}/requirements.txt ..."
# pip install --no-cache-dir -r ${APP_BASE_DIR}/requirements.txt

# log.info "> Installing ${APP_NAME}..."
# pip install -e ${APP_BASE_DIR}

log.info "> Setting up nginx config..."
cp -v "${APP_BASE_DIR}/build-resources/config/nginx.conf" "/etc/nginx/nginx.conf"
log.info "> Done Setting up nginx config"

log.info "> Removing default nginx site..."
rm -f "/etc/nginx/sites-available/default"
rm -f "/etc/nginx/sites-enabled/default"
log.info "> Done removing default nginx site"

log.info "> Chown'ing necessary files to $APPUSER ..."
touch /etc/nginx/sites-enabled/default
chown -R $APPUSER:$APPUSER /etc/nginx/sites-enabled/default /var/log/nginx /var/lib/nginx
log.info "> Done chown'ing necessary files to $APPUSER ..."

log.info "> ${APP_NAME} installation complete."

### Cleanup
log.info "> Cleaning up ${APP_NAME} build..."

log.info "> apt-get autoremove"
apt-get autoremove -y

log.info "> apt-get clean"
apt-get clean -y

log.info "> removing uncleaned apt files in '/var/lib/apt/lists/'"
rm -rf /var/lib/apt/lists/*

log.info "> remove build files from '${APP_BASE_DIR}/build'"
rm -rf "${APP_BASE_DIR}/build/"

log.info "> ${APP_NAME} build cleanup complete."

log.info "> Finished 'docker-pre-build.sh' ($(date))"
