import abc

from PyPDF2 import PdfFileWriter, PdfFileReader
from typing import List, Dict, Set, Tuple

from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput
from biomed.common.biomed_summary import FullSummaryResponse


class Bookmark(abc.ABC):
    """
    Base class that holds bookmark information and abstract methods that enable bookmark creation from category name
    ie: DiseaseDisorder and a Biomed Output object
    """

    def __init__(self,
                 title: str,
                 page: int,
                 parent_id: str = None,
                 child_ids: List[str] = None,
                 bookmark_id: str = None):
        self.title = title
        self.page = page
        self.parent_id = parent_id
        self.child_ids = child_ids or set()
        self.id = bookmark_id or title

    @classmethod
    @abc.abstractmethod
    def get_bookmark_name(cls, category_name: str, biomed_output: BiomedOutput) -> str:
        # must be implemented, this returns the name displayed in the bookmark viewer
        raise NotImplementedError

    @classmethod
    def get_id(cls, category_name: str, biomed_output: BiomedOutput, parent_id: str) -> str:
        # defaults to being unique based on bookmark name, only unique bookmark ids will get created in the document
        id_str = parent_id or ''
        if id_str:
            id_str += f'-'
        # use the bookmark name as the id for the bookmark by default
        id_str += cls.get_bookmark_name(category_name=category_name, biomed_output=biomed_output)

        return id_str

    @classmethod
    def init_from_biomed_output(cls, category_name, biomed_output: BiomedOutput, parent_id: str):
        return cls(
            page=biomed_output.page - 1,
            title=cls.get_bookmark_name(category_name=category_name, biomed_output=biomed_output),
            bookmark_id=cls.get_id(category_name=category_name, biomed_output=biomed_output,  parent_id=parent_id),
            parent_id=parent_id
        )


class BiomedCategoryBookmark(Bookmark):
    @classmethod
    def get_bookmark_name(cls, category_name: str, biomed_output: BiomedOutput) -> str:
        return category_name


class BiomedGroupBookmark(Bookmark):
    @classmethod
    def get_bookmark_name(cls, category_name: str, biomed_output: BiomedOutput) -> str:
        # uses the pref text or if no such thing exists it uses label_uncoded
        return getattr(biomed_output, 'preferredText', None) or f'{biomed_output.label}_uncoded'

    @classmethod
    def get_id(cls, category_name: str, biomed_output: BiomedOutput, parent_id: str) -> str:
        # this method returns unique IDs based on cuis rather than on the pref text that is used for the bookmark name
        # because there can be a 1:many mapping of cuis to pref texts
        id_str = parent_id or ''
        if id_str:
            id_str += f'-'

        id_str += getattr(biomed_output, 'cui', None) or biomed_output.label

        return id_str


class BiomedTextBookmark(Bookmark):
    @classmethod
    def get_bookmark_name(cls, category_name, biomed_output) -> str:
        return biomed_output.text

class BiomedPreferredTextBookmark(Bookmark):
    @classmethod
    def get_bookmark_name(cls, category_name: str, biomed_output: BiomedOutput) -> str:
        # uses the pref text or if no such thing exists it uses label_uncoded
        return getattr(biomed_output, 'preferredText', None) or biomed_output.label

class BiomedLabelBookmark(Bookmark):
    @classmethod
    def get_bookmark_name(cls, category_name, biomed_output) -> str:
        return biomed_output.label or 'ExplicitICD10Code'


# CLASS THAT CONTAINS AND ADDS BOOKMARKS, to change the hierarchy change the PARENT_BOOKMARK_CLASS setting in the
# individual bookmark classes
class BookmarkTree:
    def __init__(self, hierarchy: List[Bookmark] = None):
        self.top_level_ids = []
        self.bookmark_id_mapping: Dict[str, Bookmark] = dict()
        # hierarchy goes from top to bottom, so last entry is the lowest level
        self.hierarchy = hierarchy or [BiomedCategoryBookmark, BiomedGroupBookmark, BiomedTextBookmark]

    def add_biomed_output_bookmark(self, category_name, biomed_output: BiomedOutput):
        parent_id = ''
        # loop through the hierarchy from top to bottom
        for idx in range(len(self.hierarchy)):
            bookmark_type = self.hierarchy[idx]
            # check to see if the bookmark already exists, create the bookmark if it doesnt
            new_bookmark_id = bookmark_type.get_id(
                category_name=category_name,
                biomed_output=biomed_output,
                parent_id=parent_id)

            # if we're at the bottom level of the hierarchy add page to the id so that there are unique values
            # for every page
            if idx == len(self.hierarchy) - 1:
                new_bookmark_id += f'_{biomed_output.page}'

            if new_bookmark_id not in self.bookmark_id_mapping:
                bookmark = bookmark_type.init_from_biomed_output(
                    category_name=category_name,
                    biomed_output=biomed_output,
                    parent_id=parent_id
                )
                self.bookmark_id_mapping[new_bookmark_id] = bookmark
                # if we're not looking at a top level bookmark
                if idx != 0:
                    # add the new child as a child in parent bookmark set of child ids
                    self.bookmark_id_mapping[parent_id].child_ids.add(new_bookmark_id)
                    # update the parent bookmark page to be the first page
                    self.bookmark_id_mapping[parent_id].page = min(
                        self.bookmark_id_mapping[parent_id].page,
                        bookmark.page
                    )
                else:
                    # add the top level bookmarks that are new to the list of new bookmark ids that is used during
                    # creation
                    self.top_level_ids.append(new_bookmark_id)

            # set the parent id to be the last looped through bookmark id
            parent_id = f'{new_bookmark_id}'

    def ingest_aspect_response(self, aspect_response: AspectResponse):
        """
        :param aspect_response: AspectResponse
        :return: None, populates the bookmark tree with all biomed outputs
        """
        category_name = aspect_response.category_name
        for biomed_out in aspect_response.response_list:
            self.add_biomed_output_bookmark(category_name=category_name, biomed_output=biomed_out)

    def add_to_pdf(self, pdf_writer: PdfFileWriter):
        id_to_parent_bookmark_mapping = {}
        children_ids = self.top_level_ids
        while len(children_ids) > 0:
            latest_id = children_ids.pop(0)
            bookmark_object: Bookmark = self.bookmark_id_mapping[latest_id]
            if not bookmark_object.parent_id:
                bookmark = pdf_writer.addBookmark(title=bookmark_object.title, pagenum=bookmark_object.page, bold=True)
            else:
                bookmark = pdf_writer.addBookmark(
                    title=bookmark_object.title,
                    pagenum=bookmark_object.page,
                    parent=id_to_parent_bookmark_mapping[bookmark_object.parent_id])
            if bookmark_object.child_ids:
                id_to_parent_bookmark_mapping[bookmark_object.id] = bookmark
                children_ids.extend(bookmark_object.child_ids)
        return pdf_writer


def create_bookmarks(
        pdf_writer: PdfFileWriter,
        biomed_summary: FullSummaryResponse,
        categories_to_include: Set[str] = None,
        bookmark_hierarchy: List[Bookmark] = None,
):
    """
    :param pdf_writer: PdfFileWriter object
    :param biomed_summary: A fullSummaryResponse object
    :param categories_to_include: biomed parent categories to use for bookmarks, if none will use everything in the
    summary
    :return: PdfFileWriter object with bookmarks added
    """

    if categories_to_include:
        bookmark_categories = biomed_summary.category_names.intersection(categories_to_include)
    else:
        bookmark_categories = biomed_summary.category_names

    # create bookmark tree, to change the hierarchy of navigation pass in a list of Bookmark classes here
    if bookmark_hierarchy:
        bookmark_mapping = BookmarkTree(bookmark_hierarchy)
    else:
        bookmark_mapping = BookmarkTree()

    for aspect_response in biomed_summary.aspect_responses:
        if aspect_response.category_name in bookmark_categories:
            bookmark_mapping.ingest_aspect_response(aspect_response)

    # read bookmark tree to get boookmarks added to pdf writer class
    pdf_writer = bookmark_mapping.add_to_pdf(pdf_writer=pdf_writer)

    return pdf_writer
