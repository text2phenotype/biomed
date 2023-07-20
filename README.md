biomed
========
Python Tools for Natural Language Processing and Machine Learning, applied to the clinical data.
In order to run the quickstart example you must have a configured awsclient

It is recommended that you set up aws role credentials by following the instructions here:
https://github.com/text2phenotype/engineer-toolbox/tree/master/examples/aws

Quickstart example
--------------------

Preliminary installs for Mac OSX (>=10.14) using [`brew`](https://brew.sh/)
```
brew install python@3.8
brew install mysql
brew install git-lfs  # or apt-get install git-lfs
```

Text2phenotype developer environment installation steps:
```bash
git lfs install
git lfs pull
git clone https://github.com/text2phenotype/text2phenotype-samples.git
git clone https://github.com/text2phenotype/text2phenotype-py.git
git clone https://github.com/text2phenotype/feature-service.git
git clone https://github.com/text2phenotype/biomed.git
# add these export lines to your .bashrc/.zshrc profile:
export TEXT2PHENOTYPE_SAMPLES=/opt/text2phenotype-samples #/your/path/text2phenotype-samples
export DATA_ROOT=/opt/S3   #/your/path/amazon-S3

# create the virtual environment
cd biomed
python3 -m pip install virtualenv
python3 -m virtualenv ve --python=/usr/local/opt/python@3.8/Frameworks/Python.framework/Versions/3.8/bin/python3
source ve/bin/activate

pip install -e ../text2phenotype-py #/your/path/text2phenotype-py
pip install -e ../feature-service #/your/path/feature-service
pip install -e . -r requirements-dev.txt
# dvc remote add -d biomed-data s3://biomed-data/prod-models
dvc pull
```

**NOTE** that to compile `mysqlclient` with the correct version of openssl,
you may need to export the following environment variables
prior to running `pip install -r requirements-dev.txt`. This
will depend on the version of openssl installed by your system:
```bash
export LDFLAGS="-L/usr/local/opt/openssl@1.1/lib"
export CPPFLAGS="-I/usr/local/opt/openssl@1.1/include"
pip install -r requirements-dev.txt
```

biomed.package
-----------------
 * **aspect**: aspect labeling, example: "Medication" vs "Allergy"
 * **deid**: tag PHI for removal from clinical text
 * **demographic**: extract patient names, DOB date of birth, etc.
 * **featureset**: tokenize text and annotate with coded concepts from cTAKES and textual patterns
 * **hepc**: custom vocab and grammar support, useful template for other custom vocabs
 * **med**: deep learning medication model, depends on cTAKES DrugNER
 * **models**: deep learning model interface and model cache, all models depend on featureset
 * **mpi**: master patient index, work in progress
 * **problem**: deep learning "problem list/diagnosis", depends on cTAKES clinical
 * **summary**: combines "med", "problem", "aspect" models with cTAKES to create a high level view of most relevant coded clinical concepts


MySQL Database (optional)
---------------------------
MySQL database scripts for reading/writing biomed databases linked to UMLS (google-drive/umls)
For NACORS, these scripts are used to count patient numbers and write the paper.
For HEPC, the database is used to compile the HEPC grammar from sources.

* sql/`make_clean.sh` invoke create scripts for the PHI and DEID.
* sql/phi   patient identifiers no clinical concepts.
* sql/deid  clinical concepts no PHI.

## Biomed Service
This service uses the [Connexion](https://github.com/zalando/connexion) library on top of Flask.
In order to run biomed you must point towards a feature-service endpoint by setting MDL_COMN_FEAT_API_BASE.

## Requirements
Python >=3.6, 3.7 preferred

## Usage
To run the service, please execute the following from the root directory:

```
pip3 install -r requirements.txt
python3 -m biomed
```

and open your browser to here:

```
http://localhost:8080/ui/
```

Your OpenAPI definition lives here:

```
http://localhost:8080/openapi.json
```

## Models Metadata Service
To run the lightweight limited API provides only models metadata:

```
python3 -m biomed --models-metadata-api
```

# Docker builds

There is now support for building and testing docker containers. This work is based on the github repository [text2phenotype/build-tools](https://github.com/text2phenotype/build-tools). Please refer to the documentation in that repository for more information.
