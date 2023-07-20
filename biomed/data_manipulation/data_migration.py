import os

from text2phenotype.common import common


class SandsFilePaths:
    def __init__(self, uuid: str, parent_dir: str):
        self.uuid = uuid
        self.parent_dir = parent_dir

    @property
    def doc_folder_path(self):
        return os.path.join('processed', 'documents', self.uuid)

    @property
    def metadata_fp(self):
        return os.path.join(self.parent_dir, self.doc_folder_path, f'{self.uuid}.metadata.json')

    @property
    def metadata(self):
        if os.path.isfile(self.metadata_fp):
            return common.read_json(self.metadata_fp)
        else:
            return {}

    @property
    def job_id(self):
        return self.metadata.get('job_id')

    @property
    def text_path(self):
        return os.path.join(self.parent_dir, self.doc_folder_path, f'{self.uuid}.extracted_text.txt')

class JobMeta:
    def __init__(self, job_id, parent_dir):
        self.job_id = job_id
        self.parent_dir = parent_dir

    @property
    def job_meta_path(self):
        return os.path.join('processed/jobs', self.job_id)

    @property
    def job_meta_dict(self):
        return common.read_json(os.path.join(self.parent_dir, self.job_meta_path, f'{self.job_id}.manifest.json'))

    @property
    def app_dest(self):
        return self.job_meta_dict.get('app_destination',  'default')

    @property
    def doc_ids(self):
        return set(self.job_meta_dict.get('document_info', {}).keys())
