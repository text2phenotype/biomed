#!/usr/bin/env bash

set -e
set -o pipefail

echo "> Starting 'collect-fs-resources.sh' ($(date))"

cd $WORKSPACE/biomed/
echo "> Working directory: $(pwd)"

### Python Sanity Check
echo "> Using Python3..."
which python3
python3 --version

echo "> Upgrading virtualenv PIP..."
python3 -m pip install -U pip

echo "> Using PIP:"
which pip

echo "> These PIP versions should match!"
pip --version
pip3 --version
### Python Sanity Check

echo "> Installing DVC"
pip install dvc[s3]

echo "> DVC Pull"
time python3 -m dvc pull -f -j9 -v

echo "> Removing DVC"
pip uninstall dvc -y

# Build Docker Image with Biomed resources
cd ${WORKSPACE}/biomed/
echo "> Working directory: $(pwd)"

DOCKERFILE_NAME="Dockerfile.collect-biomed-resources"
DOCKER_IMAGE_TARGET="text2phenotype.com/biomed-resources:dev_latest"

# Create Dockerfile
cat > "${DOCKERFILE_NAME}" <<EOF
FROM alpine:latest as resources

COPY ./resources /resources

WORKDIR /resources

RUN mkdir -p ./1/files/ ./2/files/ ./3/files/ ./4/files/ ./5/files/ ./6/files/ ./7/files/ && \
    mv files/doc_type/ ./1/files/ && \
    mv files/bert_embedding/ ./2/files/ && \
    mv files/diagnosis/ ./3/files/ && \
    mv files/oncology/ ./4/files/ && \
    mv files/drug/ ./5/files/ && \
    mv files/dos/ ./6/files/ && \
    mv files/* ./7/files/


FROM alpine:latest

RUN apk add rsync

COPY --from=resources /resources/1/files/ /app/shared-content/biomed/biomed/resources/files
COPY --from=resources /resources/2/files/ /app/shared-content/biomed/biomed/resources/files
COPY --from=resources /resources/3/files/ /app/shared-content/biomed/biomed/resources/files
COPY --from=resources /resources/4/files/ /app/shared-content/biomed/biomed/resources/files
COPY --from=resources /resources/5/files/ /app/shared-content/biomed/biomed/resources/files
COPY --from=resources /resources/6/files/ /app/shared-content/biomed/biomed/resources/files
COPY --from=resources /resources/7/files/ /app/shared-content/biomed/biomed/resources/files

EOF

time docker build . \
    -f "${DOCKERFILE_NAME}" \
    -t "${DOCKER_IMAGE_TARGET}"

time docker push "${DOCKER_IMAGE_TARGET}"

echo "> Finished 'collect-fs-resources.sh' ($(date))"
