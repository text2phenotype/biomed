import os.path
from collections import defaultdict
from math import ceil

from PyPDF2 import PdfFileReader, PdfFileWriter
from text2phenotype.common import common
#
# incomplete_doc_ids = ['152ae8c64abc41f1a3da36f402b72432',
#                       '3888632066d345c6b7f678db87d1725e',
#                       '3cd82c7f738d4bb99ad6abd546391a65',
#                       '61ba87b93e9e49219b328e247765fede',
#                       '7fc42041f1e54b47b5c2f64165c9c55f',
#                       '86d0c4f2ae43481fb7e68975f4879dbf',
#                       'a4d2e2b51b0045e585d304e36ed2fa7a',
#                       'b469fc48125b4a5c99d45d6e878d39f5',
#                       'b4de743d18ff4c848bf1eb179cb8402c',
#                       'b6b694f44f894e65b5e0a4abaf775055',
#                       'bffd0236a7ba4c2085dc9a3c5ed61b2f',
#                       'e14cab1391894c1591fecf92c2e6f9f2',
#                       'e1bc4609180b4ba3a016a74e444ff24b',
#                       'eb45b96a404b4d81b5f7cd7187c38d4e',
#                       'f01f13f97b934d7fb4e68ac9fa607949',
#                       'f07e5c5d57ab4dd88853323defecedd0']
#
# job_meta_path = '/Users/shannonfee/Documents/hcc_coding_pilot/processed/b222ca6d06cc4f2cb5381dc1374d4c7b.manifest.json'
# job_meta = common.read_json(job_meta_path)
pdf_paths = common.get_file_list('/Users/shannonfee/Downloads/UC_Batch_1', '.pdf', True)
out_dir = '/Users/shannonfee/Downloads/UC_Batch_1_Chunks'
os.makedirs(out_dir, exist_ok=True)
CHUNK_SIZE = 5
for orig_pdf in pdf_paths:
    if os.path.isfile(orig_pdf):
        with open(orig_pdf, 'rb') as infile:
            pdf_reader = PdfFileReader(infile)
            num_chunks = ceil(pdf_reader.getNumPages() / CHUNK_SIZE)
            for chunk in range(num_chunks):
                pdf_writer = PdfFileWriter()
                for page in range(chunk * CHUNK_SIZE, min((chunk + 1) * CHUNK_SIZE, pdf_reader.getNumPages())):
                    pdf_writer.addPage(pdf_reader.getPage(page))
                pdf_name = os.path.join(out_dir, os.path.basename(orig_pdf).replace('.pdf', f'_chunk_{chunk}.pdf'))
                with open(pdf_name, 'wb') as outfile:
                    pdf_writer.write(outfile)
    else:
        print(orig_pdf)



# file_mapping = defaultdict(list)
# for i in sorted(common.get_file_list('/Users/shannonfee/Documents/CodingPilotChunkOutput', '.pdf', True)):
#     file_basename = os.path.basename(i)
#     base_file = file_basename.split('_')[0]
#     file_mapping[base_file].append(i)

# out_dir = '/Users/shannonfee/Documents/CodingPilotChunkRecombined'
# os.makedirs(out_dir, exist_ok=True)
# for doc, subfiles in file_mapping.items():
#     pdf_writer= PdfFileWriter()
#     for subfile in subfiles:
#         pdf_reader = PdfFileReader(open(subfile, 'rb'))
#         for page in range(pdf_reader.getNumPages()):
#             pdf_writer.addPage(pdf_reader.getPage(page))
#     outfile = os.path.join(out_dir, f'{doc}.pdf')
#     with open(outfile, 'wb') as out_stream:
#         pdf_writer.write(out_stream)