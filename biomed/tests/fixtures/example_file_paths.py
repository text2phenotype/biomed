import os

this_dir=os.path.dirname(__file__)
files_dir = os.path.join(this_dir, "ingest_example_files")
david_vaughan_files_dir = os.path.join(files_dir, 'david_vaughan')
stephan_garcia_files_dir = os.path.join(files_dir, "stephan_garcia")
tina_mormol_files_dir = os.path.join(files_dir, "tina_mormol")
carolyn_blose_files_dir = os.path.join(files_dir, "carolyn_blose")
john_stevens_files_dir = os.path.join(files_dir, "john_stevens")

# John Stevens ------
john_stevens_pdf_filepath = os.path.join(john_stevens_files_dir, "john-stevens.pdf")
john_stevens_txt_filepath = os.path.join(john_stevens_files_dir, "john-stevens.pdf.txt")


# David Vaughan ------
david_vaughan_pdf_filepath = os.path.join(david_vaughan_files_dir, "david-vaughan.pdf")
david_vaughan_txt_filepath = os.path.join(david_vaughan_files_dir, "david-vaughan.pdf.txt")

# Stephan Garcia ------
stephan_garcia_pdf_filepath = os.path.join(stephan_garcia_files_dir, "stephan-garcia.pdf")
stephan_garcia_txt_filepath = os.path.join(stephan_garcia_files_dir, "stephan-garcia.pdf.txt")

# Tina Mormol ------
tina_mormol_pdf_filepath = os.path.join(tina_mormol_files_dir, "tina-mormol.pdf")
tina_mormol_txt_filepath = os.path.join(tina_mormol_files_dir, "tina-mormol.pdf.txt")

# Carolyn Blose ------
carolyn_blose_pdf_filepath = os.path.join(carolyn_blose_files_dir, "carolyn-blose.pdf")
carolyn_blose_txt_filepath = os.path.join(carolyn_blose_files_dir, "carolyn-blose.pdf.txt")
carolyn_blose_phi_tokens_json_filepath = os.path.join(carolyn_blose_files_dir, "carolyn-blose.pdf.txt.phi_tokens.json")

# empty text filepath
empty_text_fp = os.path.join(files_dir, 'empty_record.txt')


# working_dir_filepath
uuid = 'f4752da314ac447287ad3a4a7d72e7fe'
working_dir = os.path.join(this_dir, 'sample_working_bucket_files', uuid)
working_clin_summary_fp = os.path.join(working_dir, f'{uuid}.clinical_summary.json')
text_coords_fp = os.path.join(working_dir, f'{uuid}.text_coordinates')
source_pdf_fp = os.path.join(working_dir, f'{uuid}.source.pdf')

sample_image_path = os.path.join(working_dir, 'pages', f'{uuid}.page_0001.png')