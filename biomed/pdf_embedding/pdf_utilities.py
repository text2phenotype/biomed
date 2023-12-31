from math import cos, sin, pi
from typing import List, Tuple

from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.generic import DictionaryObject, NumberObject, NameObject, ArrayObject, FloatObject, NullObject, \
    IndirectObject
from PyPDF2.pdf import PageObject

from text2phenotype.annotations.file_helpers import TextCoordinate


def read_and_get_full_pdf_writer(pdf_file_path) -> PdfFileWriter:
    """
    :param pdf_file_path: path to pdf you want to edit
    :return: a pdf writer with all pages from pdf you want to edit already populated
    """
    infile = open(pdf_file_path, 'rb')
    pdf_reader = PdfFileReader(infile)
    pdf_writer = PdfFileWriter()
    for page in range(pdf_reader.numPages):
        pdf_writer.addPage(pdf_reader.getPage(page))
    return pdf_writer


def get_pdf_page_mapping(pdf_writer: PdfFileWriter) -> List[dict]:
    pg_output = []
    for pg_no in range(pdf_writer.getNumPages()):
        page = pdf_writer.getPage(pg_no)
        width, height = page.mediaBox[2], page.mediaBox[3]
        pg_output.append({'width': width, 'height': height, 'page': page})
    return pg_output


def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height


def update_image_to_pdf_coords(
        text_coords: TextCoordinate,
        pdf_width: int,
        pdf_height: int,
        page_image_width: int,
        page_image_height: int):
    """
    :param text_coords: TextCoordinate object, contains png coordinates for a given text (token).
     Note that the  ocr text tokens do  not necessarily  have to =  our definitionof a token
    :param pdf_width: integer that corresponds to the width of the visible portion of the pdfPage
    assumes that  the page media box (aka visible section  has one corner at 0, 0 and is a rectangle)
    :param pdf_height: integer that corresponds to the height of the visible portion of the pdfPage
    assumes that  the page media box (aka visible section  has one corner at 0, 0 and is a rectangle)
    :param page_image_width: the image width in pixels
    :param page_image_height: the image height in pixels
    :return: updated coordinates that correspond to pdf positioning
    scales horizontal and scales and invertss vertical axis
    """
    x1_new = text_coords.left * pdf_width / page_image_width
    x2_new = text_coords.right * pdf_width / page_image_width
    y1_new = pdf_height - text_coords.bottom * pdf_height / page_image_height
    y2_new = pdf_height - text_coords.top * pdf_height / page_image_height
    return x1_new, y1_new, x2_new, y2_new


def rotate_coords(x1, y1, page_height, page_width, rotation_angle):
    if isinstance(rotation_angle, IndirectObject):
        rotation_angle = rotation_angle.getObject()
    rot_angle = rotation_angle * pi/180
    page_center_x = page_width / 2
    page_center_y = page_height / 2
    x1_updated = (x1 - page_center_x) * cos(rot_angle) - (y1 - page_center_y) * sin(rot_angle) + page_center_x
    y1_updated = (x1 - page_center_x) * sin(rot_angle) + (y1 - page_center_y) * cos(rot_angle) + page_center_y
    return x1_updated, y1_updated


def create_highlight(x1, y1, x2, y2, color: List[int]):
    """
    :param x1, y1: pdf coordinates for bottom left corner
    :param x2, y2: pdf coordinates for top right corner
    :param color: 3 digit list of floats between 0, 1 that corresponds to a color
    :return: a pdfHighlight object
    basically  copied with minor adjustments from
    https://stackoverflow.com/questions/7605577/read-highlight-save-pdf-programmatically
    """
    new_highlight = DictionaryObject()

    new_highlight.update({
        NameObject("/F"): NumberObject(4),
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Highlight"),

        NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
        NameObject("/Rect"): ArrayObject([
            FloatObject(x1),
            FloatObject(y1),
            FloatObject(x2),
            FloatObject(y2)
        ]),
        NameObject("/QuadPoints"): ArrayObject([
            FloatObject(x1),
            FloatObject(y2),
            FloatObject(x2),
            FloatObject(y2),
            FloatObject(x1),
            FloatObject(y1),
            FloatObject(x2),
            FloatObject(y1)
        ]),
    })

    return new_highlight


def add_highlight_to_page(highlight, page, pdf_writer: PdfFileWriter) -> Tuple[PdfFileWriter, object]:
    """
    Add a highlight to a PDF page.

    Parameters
    ----------
    highlight : Highlight object
    page : PDF page object
    pdf_writer : PdfFileWriter object
    :returns a tuple of the file_writer class, and a reference to the page. Both should be edited in
     place and shouldn't require this but due to the nesting of functions if we don't return these values the many
      highlights don't get attached
    copied with minor edits from https://stackoverflow.com/questions/7605577/read-highlight-save-pdf-programmatically
    """
    highlight_ref = pdf_writer._addObject(highlight)

    if "/Annots" in page and not isinstance(page['/Annots'], NullObject):
        page[NameObject("/Annots")].append(highlight_ref)
    else:
        page[NameObject("/Annots")] = ArrayObject([highlight_ref])
    return pdf_writer, page


def add_single_highlight_from_text_coords(
        text_coords: TextCoordinate,
        pdf_page_width: int,
        pdf_page_height: int,
        img_height: int,
        img_width: int,
        pdf_writer: PdfFileWriter,
        pdf_page: PageObject,
        color: List[float]
):
    """
    :param text_coords: a TextCoordinate object
    :param pdf_page_width: pdf page width (width of pdf media box)
    :param pdf_page_height: pdf page height (height of  media box)
    :param img_height: image used for ocr'ing the page, height in pixels
    :param img_width: image used for ocr'ing the page,  width in pixels
    :param pdf_writer: the pdffilewriter object
    :param pdf_page: a page of a pdf. result of  pdffilewriter.getpage(page_no)
    :param color: the color of the highlight, 3 digit list of floats between 0, 1
    :return: the updated pdf file writer and pdf  page after adding  the highlight
    """
    x1, y1, x2, y2 = update_image_to_pdf_coords(
        text_coords=text_coords,
        pdf_width=pdf_page_width,
        pdf_height=pdf_page_height,
        page_image_width=img_width,
        page_image_height=img_height
    )
    if pdf_page.get('/Rotate'):
        x1, y1 = rotate_coords(
            x1, y1,
            page_height=pdf_page_height,
            page_width=pdf_page_width,
            rotation_angle=pdf_page.get('/Rotate')
        )
        x2, y2 = rotate_coords(
            x2, y2,
            page_height=pdf_page_height,
            page_width=pdf_page_width,
            rotation_angle=pdf_page.get('/Rotate')
        )

    highlight = create_highlight(x1, y1, x2, y2, color=color)
    pdf_writer, pdf_page = add_highlight_to_page(highlight=highlight, page=pdf_page, pdf_writer=pdf_writer)
    return pdf_writer, pdf_page
