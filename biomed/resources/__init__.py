import os

from biomed.biomed_env import BiomedEnv


# DEVOPS-2312 removes DVC pull from the build steps, container will not have resources.
# Now all resources are stored in the SSS, location passed in the MDL_BIOM_MODELS_PATH env var.
LOCAL_FILES = os.path.join(BiomedEnv.BIOM_MODELS_PATH.value, 'resources/files')

CUI_RULE = os.path.join(LOCAL_FILES, 'concept_aspect_map.json')
TUI_RULE = os.path.join(LOCAL_FILES, 'semtype_aspect_map.json')
HEADING_FILE = os.path.join(LOCAL_FILES, 'headings', 'header_info.json')
