# syntax=docker/dockerfile:experimental
# Use a reasonable FROM image
ARG IMAGE_FROM_TAG
FROM text2phenotype.com/text2phenotype-py:${IMAGE_FROM_TAG:-latest}

# Create a list of build arguments
ARG APP_ENVIRONMENT
ARG APP_GIT_SHA
ARG APP_SERVICE
ARG IMAGE_FROM_TAG

# Set environment variables
# UNIVERSE_IS_VERBOSE enables log level INFO.
ENV UNIVERSE_IS_VERBOSE=true

### Application metadata
ENV APP_ENVIRONMENT="${APP_ENVIRONMENT:-prod}"
ENV APP_GIT_SHA="${APP_GIT_SHA:-unset}"
ENV APP_NAME="Biomed"
ENV IMAGE_FROM_TAG="${IMAGE_FROM_TAG}"

# APP_SERVICE should be one of:
# biomed-api
# worker-deid
# worker-demographics
# worker-summary-clinical
# worker-summary-oncology
# worker-oncology-only
# worker-phi-token
ENV APP_SERVICE="${APP_SERVICE:-biomed-api}"

### File path locations
ENV APP_BASE_DIR="/app"
ENV PATH="${APP_BASE_DIR}/bin/:${PATH}"

# Set some container options
WORKDIR "${APP_BASE_DIR}"
EXPOSE 8080

###  This section should be ordered in such a way that the least likely
### operation to change should be first.

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
  mkdir -p /var/lib/apt/lists/; \
  /usr/bin/flock -w 900 -F /var/lib/apt/lists/lock \
    /usr/bin/apt-get update && \
    apt-get install -y default-libmysqlclient-dev libpng-dev nginx gfortran libatlas-base-dev

# Models are now in shared storage
# COPY --chown=5001:5001 ./resources "${APP_BASE_DIR}"/biomed/resources/
# COPY --chown=5001:5001 ./requirements.txt "${APP_BASE_DIR}"/
# RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY --chown=5001:5001 ./ "${APP_BASE_DIR}"/
# COPY --chown=5001:5001 ./build-resources/ "${APP_BASE_DIR}"/build-resources/
# COPY --chown=5001:5001 ./build-tools/bin/ "${APP_BASE_DIR}"/bin/
# COPY --chown=5001:5001 *.py *.yaml *.txt *.json Jenkins* .docker.metadata "${APP_BASE_DIR}"/
# COPY --chown=5001:5001 ./tests "${APP_BASE_DIR}"/tests/

RUN  mv ${APP_BASE_DIR}/build-resources/bin/* "${APP_BASE_DIR}/bin/" && \
     mv ${APP_BASE_DIR}/build-tools/bin/* "${APP_BASE_DIR}"/bin/ && \
     "${APP_BASE_DIR}/bin/docker-pre-build.sh"

# Copy the application code last and pip install it

RUN --mount=type=cache,target=/root/.cache pip install -e ${APP_BASE_DIR} && \
    chown -R 5001:5001 "${APP_BASE_DIR}/biomed.egg-info/"

USER 5001

# dumb-init is used to assist with proper signal handling, without
# it we will not kill the other processes
ENTRYPOINT ["/usr/local/bin/dumb-init","--rewrite","15:30","--"]

# This command is what launches the service by default.
CMD ["/bin/bash", "-c", "${APP_BASE_DIR}/bin/startup.sh"]
