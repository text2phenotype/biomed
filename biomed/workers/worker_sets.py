from biomed.workers.covid_lab.worker import CovidLabTaskWorker
from biomed.workers.deid.worker import DeidTaskWorker
from biomed.workers.demographics.worker import DemographicsTaskWorker
from biomed.workers.device_procedure.worker import DeviceProcedureTaskWorker
from biomed.workers.disease_sign.worker import DiseaseSignTaskWorker
from biomed.workers.doc_type.worker import DocumentTypeTaskWorker
from biomed.workers.drug.worker import DrugModelTaskWorker
from biomed.workers.imaging_finding.worker import ImagingFindingTaskWorker
from biomed.workers.lab.worker import LabModelTaskWorker
from biomed.workers.oncology_only.worker import OncologyOnlyTaskWorker
from biomed.workers.phi_token.worker import PHITokenWorker
from biomed.workers.smoking.worker import SmokingTaskWorker
from biomed.workers.vital_sign.worker import VitalSignTaskWorker
from biomed.workers.summary.worker import SummaryWorker

SINGLE_MODEL_WORKERS = {DrugModelTaskWorker, CovidLabTaskWorker, DemographicsTaskWorker, DeviceProcedureTaskWorker,
                        DiseaseSignTaskWorker, DocumentTypeTaskWorker, ImagingFindingTaskWorker, LabModelTaskWorker,
                        OncologyOnlyTaskWorker, PHITokenWorker, SmokingTaskWorker, VitalSignTaskWorker}


SUMMARY_MODEL_WORKERS = {SummaryWorker}


OTHER_BIOMED_WORKERS = {DeidTaskWorker}
