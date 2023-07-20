import argparse
import os

from biomed.pdf_embedding.pdf_utilities import get_num_pixels

from biomed.pdf_embedding.create_pdf_highlight import embed_write_pdf
from biomed.summary.text_to_summary import get_page_indices
from biomed.tests.experiment.post_processing.finding_files import get_job_uuids, \
    get_text_coord_path, get_pdf_path, get_biomed_json_files, get_working_dir_file
from text2phenotype.annotations.file_helpers import TextCoordinateSet
from biomed.common.biomed_summary import combine_all_biomed_output_fps
from text2phenotype.common import common
from biomed.pdf_embedding.pdf_bookmarking import *

# to change the hierarchy of bookmarks pass in a list of Bookmark classes,
BOOKMARK_HIERARCHY = [BiomedCategoryBookmark, BiomedPreferredTextBookmark, BiomedLabelBookmark, BiomedTextBookmark]
# see the BookmarkTree class for more details

"""
To Run this Script you Need the Working folder (not the outbox folder) for all documents in a job
    To do this I 
    1. pull the job manifest from the job (from the outbox folder bc its easier to find there)
    2. get the uuids from the job manifest as 
        print(" ".join(list(common.read_json(job_manifest_fp)['document_info'].keys())))
    3. copy the output from the output line [DOC ID STR]
    4. Run 
        for i in [DOC_ID_STR]; 
        do aws s3 sync s3://{YOUR_BUCKET_NAME}/processed/documents/$i {LOCAL_WORKING_DIR}/$i --exclude "*chunks*";
        done
THEN Run this script with the parameters -job_manifest_path = local path ot job manifest you downloaded, 
-local_working_dir {LOCAL_WORKING_DIR}, -biomed suffixes biomed suffixes you want included (comma separated), 
-pdf_output_dir a dir you want the output in 
    # SAMPLE ARGS
    # -job_manifest_path
    # /Users/shannonfee/Documents/arkos/outbox/2021-06-03/processed/jobs/84be968fa6024a508713344172204087/84be968fa6024a508713344172204087.manifest.json
    # -local_working_dir
    # /Users/shannonfee/Documents/arkos/work/2021-06-03/processed/documents
    # -biomed_suffixes
    # clinical_summary,vital_signs
    # -pdf_output_dir
    # /Users/shannonfee/Documents/experiment_today

"""

import sys
sys.setrecursionlimit(150000)

def parse_arguments():
    """
    this parses the argument passed into the terminal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-job_manifest_path', type=str, help='Full path to local job manifest if available, otherwise specify uuid')
    parser.add_argument('-uuid', type=str, help='uuid for the document')
    parser.add_argument('-biomed_suffixes', type=str,
                        help='Which biomed operations to include in the embedded pdf results, '
                             'you cannot use demographics, must be comma seperate')
    parser.add_argument('-local_working_dir', type=str,
                        help='Local Location for File Storage, '
                             'assumes that the working directory for all uuids in the job will be at  '
                             'this_dir/{doc_uuid}/')
    parser.add_argument('-pdf_output_dir', type=str, help='Where the embedded PDF will be written')
    parser.add_argument('-categories_to_include', type=str, help='Which aspect categories to include in bookmarks, comma separated')
    parser.add_argument('--use_original_file_name', action='store_true',
                        help= 'When using job_manifest_path if true sets the output file name to be the oriignal file name')
    return parser.parse_args()


def main(args):
    if args.job_manifest_path:
        uuids = get_job_uuids(args.job_manifest_path)
    else:
        uuids = [args.uuid]
    os.makedirs(args.pdf_output_dir, exist_ok=True)
    for uuid in uuids:
        text_coord_path = get_text_coord_path(args.local_working_dir, uuid)
        if not text_coord_path or not os.path.isfile(text_coord_path):
            continue
        text_coord_set = TextCoordinateSet().fill_coordinates_from_stream(open(text_coord_path, 'rb'))
        text = common.read_text(get_working_dir_file(args.local_working_dir, uuid, 'extracted_text.txt'))
        page_numbers = get_page_indices(text)
        text_coord_set.update_from_page_ranges(page_numbers=page_numbers)
        biomed_paths = get_biomed_json_files(args.local_working_dir, uuid, args.biomed_suffixes.split(','))
        full_biomed_summary = combine_all_biomed_output_fps(biomed_paths)

        # get image_mapping
        image_fps = common.get_file_list(os.path.join(args.local_working_dir, uuid, 'pages'), '.png', True)
        image_mapping = [{}] * len(image_fps)
        for image_fp in image_fps:
            page_no: int = int(os.path.basename(image_fp).split('.')[1].split('_')[1]) - 1
            width, height = get_num_pixels(image_fp)
            image_mapping[page_no] = {'width': width, 'height': height}

        # categories to include
        if args.categories_to_include:
            categories_to_include = set(args.categories_to_include.split(','))
        else:
            categories_to_include = None
        if args.use_original_file_name and args.job_manifest_path:
            output_fn = os.path.join(
                args.pdf_output_dir,
                os.path.basename(common.read_json(args.job_manifest_path)['document_info'][uuid]['filename']))
        else:
            output_fn = os.path.join(args.pdf_output_dir, f'{uuid}.embedded.pdf')
        res = embed_write_pdf(
            source_pdf_path=get_pdf_path(args.local_working_dir, uuid),
            output_pdf_path=output_fn,
            text_coord_set=text_coord_set,
            biomed_summary=full_biomed_summary,
            image_dimensions=image_mapping,
            bookmark_hierarchy=BOOKMARK_HIERARCHY,
            categories_to_include=categories_to_include
        )

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
