Biomed workers
========

There are three types of Worker:
* **SummaryTaskWorker**
* **SingleModelBaseWorker**
* **Reassembler**

### SummaryTaskWorker (_Document_)
This worker uses source test of the document the results of any SingleModel worker (_SingleModelBaseWorker_) 
and aggregates them. 

Each worker uses a special `text2phenotype.tasks.task_info.BiomedSummaryTaskInfo` where field `model_dependencies` is defined. 
The field defines the results of which models to use.

Example schema based on Clinical Summary worker, which uses following model worker results
* `disease_sign` 
* `drug` 
* `lab` 
* `smoking`

![Summary worker schema](_doc/summary_schema.svg)

### SingleModelBaseWorker (_Chunk_)
This worker uses source text of the chunk and results of `annotate` and `vectorize` worker.

Each worker uses special `text2phenotype.tasks.task_info.SingleModelTaskInfo` and map `MODEL_TYPE_TO_VERSION_FILE_LIST` 
from _biomed/constants/constants.py_ for defining which TensorFlow models to use for prediction

Example schema based on PHI token worker, which uses `Deid` model

![Model worker schema](_doc/model_schema.svg)

### Reassembler
This worker takes all chunk's results for the document, combines them and puts results to the S3

Example schema based on Clinical Summary operation

![Reassembler worker schema](_doc/reassembler_chema.svg)

Quickstart example
--------------------

Workers require connection to RMQ 

Run worker very easy

The environment must be installed first.
This is described in the [main README.md file](../../README.md)

```
cd biomed/workers/clinical_summary
./start_worker.py
```

List of workers
-----------------

##### SummaryTaskWorker
 * **all_model_summary_worker**
 * **clinical_summary**
 * **covid_specific**
 * **oncology_summary**
 
##### SingleModelBaseWorker
 * **covid_lab** 
 * **demographics**
 * **device_procedure**
 * **disease_sign**
 * **doctype**
 * **drug**
 * **imaging_finding**
 * **lab**
 * **oncology_only**
 * **phi_tokens**
 * **smoking**
 * **vital_signs**

##### Reassemble
 * **reassembler worker** 
