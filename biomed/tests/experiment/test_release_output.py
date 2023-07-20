"""
Evaluate the outputs from the Biomed NLP release staging outputs
"""
import unittest
import os
from typing import List

from text2phenotype.common import common
from text2phenotype.constants.common import VERSION_INFO_KEY

# step 1
# sync the rwd-dev outbox folder to local output folder with aws-cli
#
#    aws s3 sync s3://v16.7.00-RC01 /opt/release_output/rwd-dev-nlp-qa/v16.7.00-RC01

# step 2
# uncomment the OUTPUT_DIR line below to test the output from dev-nlp-stage and run this script (eg in pycharm)
#
OUTPUT_DIR = "/opt/release_output/rwd-dev-nlp-qa/v16.7.00-RC01"

# step 3
# sync the mdl-phi-stage outbox folder to local output folder with aws-cli
#
#     aws s3 sync s3://prod-nlp-stage-us-east-1/outbox/tests/v16.7.00-00 /opt/release_output/rwd-prod-nlp-stage/v16.7.00

# step 4
# uncomment the OUTPUT_DIR line below to test the output from mdl-phi-stage and run this script (eg in pycharm)
#
# OUTPUT_DIR = "/opt/release_output/rwd-prod-nlp-stage/v16.6.00"


class TestJobMetadata(unittest.TestCase):
    JOB_PATH = "processed/jobs"
    JOB_MANIFEST_PATH = common.get_file_list(os.path.join(OUTPUT_DIR, JOB_PATH), ".manifest.json", True)[0]
    JOB_MANIFEST_JSON = common.read_json(JOB_MANIFEST_PATH)

    def test_keys(self):
        expected_key = {
            "version", "started_at", "completed_at", "work_type", "job_id", "document_info",
            "legacy_document_ids", "operations", "required_features", "bulk_source_bucket",
            "bulk_destination_directory", "bulk_source_directory", "user_info", "app_destination",
            "biomed_version", "total_duration", "reprocess_options", "model_version",
            "user_canceled", "stop_documents_on_failure", "summary_tasks", "user_actions_log",
            "deid_filter",
        }
        self.assertSetEqual(expected_key, set(self.JOB_MANIFEST_JSON.keys()))

    def test_total_duration(self):
        self.assertTrue(isinstance(self.JOB_MANIFEST_JSON["total_duration"], float))
        self.assertGreaterEqual(self.JOB_MANIFEST_JSON["total_duration"], 0)

    def test_string_format(self):
        string_output_keys = [
            "app_destination", "biomed_version", "bulk_source_bucket", "bulk_destination_directory",
            "bulk_source_directory", "version", "started_at", "completed_at", "work_type", "job_id",
        ]
        for key in string_output_keys:
            self.assertIsInstance(self.JOB_MANIFEST_JSON[key], str, key)

    def test_document_info(self):
        for key, value in self.JOB_MANIFEST_JSON["document_info"].items():
            self.assertIsInstance(key, str, key)
            self.assertEqual(value["status"], "completed - success", key)

    def test_operations(self):
        self.assertIsInstance(self.JOB_MANIFEST_JSON["operations"], List)
        expected_operations = {
            "genetics", "deid", "disease_sign", "clinical_summary", "covid_specific", "imaging_finding",
            "pdf_embedder", "oncology_summary", "family_history", "date_of_service", "icd10_diagnosis",
            "app_ingest", "oncology_only", "covid_lab", "phi_tokens", "demographics", "device_procedure",
            "vital_signs", "doctype", "smoking", "lab", "drug", "summary_bladder", "sdoh",
            # "summary_custom",  # MJP: to add this when we create a custom operation for release testing
        }
        self.assertSetEqual(expected_operations, set(self.JOB_MANIFEST_JSON["operations"]))

    def test_summary_tasks(self):
        self.assertIsInstance(self.JOB_MANIFEST_JSON["summary_tasks"], List)


class TestDocumentOutput(unittest.TestCase):
    DOCUMENT_PATH = "processed/documents"
    DOC_METAS = common.get_file_list(os.path.join(OUTPUT_DIR, DOCUMENT_PATH), ".metadata.json", True)
    EXTENSION_TO_TYPE_HEADERS = {
        "clinical_summary": ["DiseaseDisorder", "SignSymptom", "Smoking", "Medication", "Allergy", "Lab"],
        "covid_lab": ["CovidLabs"],
        "covid_specific": ["CovidLabs", "Device/Procedure", "Findings", "DiagnosticImaging"],
        "device_procedure": ["Device/Procedure"],
        "disease_sign": ["DiseaseDisorder", "SignSymptom"],
        "drug": ["Medication", "Allergy"],
        "imaging_finding": ["DiagnosticImaging", "Findings"],
        "lab": ["Lab"],
        "oncology_only": ["Cancer"],
        "oncology_summary": ["Medication", "Allergy", "DiseaseDisorder", "SignSymptom", "Cancer"],
        "phi_tokens": ["PHI"],
        "smoking": ["Smoking"],
        "vital_signs": ["VitalSigns"],
        "demographics": [
            "ssn", "mrn", "sex", "dob", "pat_first", "pat_last", "pat_age",
            "pat_street", "pat_zip", "pat_city", "pat_state", "pat_phone",
            "pat_email", "insurance", "facility_name", "dr_first", "dr_last",
            "pat_full_name", "dr_full_names", "race", "ethnicity",
        ],
        "sdoh": ["SocialRiskFactors"],
    }
    SINGLE_DEM_TYPES = ["pat_first", "pat_last", "dob", "sex", "pat_full_name"]
    EXPECTED_VERSION_INFO_KEYS = {
        "product_id",
        "product_version",
        "tags",
        "active_branch",
        "commit_id",
        "docker_image",
        "commit_date",
        "model_versions",
    }
    EXTENSION_TO_MODEL_TYPES_IN_VERSION_INFO = {
        "clinical_summary": {"diagnosis", "smoking", "drug", "lab"},
        "covid_lab": {"covid_lab"},
        "covid_specific": {"covid_lab", "device_procedure", "imaging_finding"},
        "device_procedure": {"device_procedure"},
        "disease_sign": {"diagnosis"},
        "drug": {"drug"},
        "imaging_finding": {"imaging_finding"},
        "lab": {"lab"},
        "oncology_only": {"oncology"},
        "oncology_summary": {"drug", "diagnosis", "oncology"},
        "phi_tokens": {"deid"},
        "smoking": {"smoking"},
        "vital_signs": {"vital_signs"},
        "demographics": {"demographic"},
        "sdoh": {"sdoh"},
    }

    def asssert_metadata_key(self, metadata: dict):
        expected_doc_metadata_keys = {
            "started_at", "completed_at", "document_info", "operations",
            "operation_results", "job_id", "total_duration",
        }
        self.assertSetEqual(expected_doc_metadata_keys, set(metadata.keys()))

    def assert_doc_info_keys(self, document_info: dict):
        expected_keys = {"document_id", "source", "source_hash", "tid"}
        self.assertSetEqual(expected_keys, set(document_info.keys()))

    def assert_operation_results(self, operation_results: dict):
        for key in operation_results:
            operation_res = operation_results[key]
            expected_keys = {"operation", "status", "results_file_key", "error_messages"}
            self.assertSetEqual(
                expected_keys, set(operation_res), f"operation {key} did not have expected keys"
            )
            self.assertEqual(
                operation_res["status"],
                "completed - success",
                f"operation {key} does not say it completed successfully",
            )
            if operation_res["results_file_key"]:
                self.assertIn("outbox/tests", operation_res["results_file_key"])
            self.assertListEqual(
                operation_res["error_messages"], [], f"operation {key} had >0 error messages {operation_res}"
            )

    def test_metadata_output(self):
        for metadata in self.DOC_METAS:
            meta_json = common.read_json(metadata)
            self.asssert_metadata_key(meta_json)
            self.assert_operation_results(meta_json["operation_results"])
            self.assert_doc_info_keys(meta_json["document_info"])

    def assert_version_info(self, version_info, operation):
        if operation != "demographics":
            self.assertEqual(len(version_info), 1)
            ver_info = version_info[0]
        else:
            self.assertIsInstance(version_info, dict)
            ver_info = version_info

        self.assertSetEqual(set(ver_info.keys()), self.EXPECTED_VERSION_INFO_KEYS)
        self.assertSetEqual(
            set(ver_info["model_versions"]), self.EXTENSION_TO_MODEL_TYPES_IN_VERSION_INFO[operation]
        )

    def test_biomed_outputs(self):
        for meta_fp in self.DOC_METAS:
            for operation in self.EXTENSION_TO_TYPE_HEADERS:
                operation_file_path = meta_fp.replace("metadata", operation)
                self.assertTrue(
                    os.path.isfile(operation_file_path),
                    f"{meta_fp} does not have a matching file for operation {operation}",
                )
                operation_output = common.read_json(operation_file_path)
                for type_header in self.EXTENSION_TO_TYPE_HEADERS[operation]:
                    self.assertIn(type_header, operation_output)
                    self.assertIsInstance(operation_output[type_header], list)
                    if operation != "demographics":
                        self.assert_proper_biomed_response(operation_output[type_header])
                    else:
                        self.assert_demographic_response(operation_output[type_header])
                        if type_header in self.SINGLE_DEM_TYPES:
                            self.assertLessEqual(
                                len(operation_output[type_header]),
                                1,
                                f"multiple values returned for {type_header}, {operation_output[type_header]}",
                            )
                self.assert_version_info(operation_output[VERSION_INFO_KEY], operation)
                self.assertEqual(
                    len(operation_output.keys()),
                    len(self.EXTENSION_TO_TYPE_HEADERS[operation]) + 1,
                    operation,
                )

    def assert_demographic_response(self, entries: List[list]):
        for entry in entries:
            self.assertEqual(len(entry), 2)
            self.assertIsInstance(entry[0], str)
            if entry[1] != 0:
                self.assertIsInstance(entry[1], float)
            self.assertGreaterEqual(entry[1], 0)
            self.assertLessEqual(entry[1], 1)

    def assert_proper_biomed_response(self, entries: List[dict]):
        for entry in entries:
            self.assertIsInstance(entry["text"], str, entry)
            self.assertIsInstance(entry["range"], list)
            self.assertEqual(len(entry["range"]), 2)
            self.assertEqual(len(entry["text"]), entry["range"][1] - entry["range"][0])
            if entry["score"] == 0:
                print(entry)
            else:
                self.assertIsInstance(entry["score"], float, entry)
                self.assertGreaterEqual(entry["score"], 0)
                self.assertLessEqual(entry["score"], 1)
            self.assertIsInstance(entry["label"], str, entry)
            other_string_types = [
                "polarity", "cui", "code", "vocab", "preferredText", "date", "tui",
                "T", "N", "M", "clinical",
            ]
            for key in entry:
                if key not in ["text", "label", "score", "range", "page"]:
                    if key in other_string_types:
                        if entry[key] is not None:
                            self.assertIsInstance(entry[key], str, key)
                    elif key in ["tty"]:
                        if entry[key] is not None:
                            self.assertIsInstance(entry[key], list, key)
                    else:
                        self.assertIsInstance(entry[key], list, key)
                        if len(entry[key]) > 0:
                            self.assertTrue(
                                isinstance(entry[key][0], str) or isinstance(entry[key][0], float)
                            )
                            self.assertIsInstance(entry[key][1], int)
                            self.assertIsInstance(entry[key][2], int)


if __name__ == "__main__":
    unittest.main()
