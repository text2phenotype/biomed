from typing import List
import math
import numpy as np
import os.path

from biomed import RESULTS_PATH

from feature_service.features.loinc import sections

from text2phenotype.common.common import write_json, read_json


class DocumentSectionizer(object):
    def __init__(self, document_list: List[str], model_file_name: None):
        self.document_types = document_list  # a list of all different document types
        self.initial_states_prob = [0.1] * len(self.document_types)  # number of doc types
        self.trans_prob = [[0.1] * len(self.document_types) for _ in range(len(self.document_types))]
        self.emission_prob = [[0.1] * len(sections) for _ in range(len(self.document_types))]
        self.model_file_name = model_file_name

    def train(self, train_section_list: List[List[int]], train_doc_list: List[List[int]]):
        # section integer is the index of the header loinc code in self.section_headers
        # doc integer is the index in document type (after translation if necessary) index in self.document_types
        # e.g, train_section_list: number of (records) X number of headers in the example
        # train_doc_list: number of examples (records) X number of headesr in the examples
        if len(train_section_list) != len(train_doc_list):
            raise ValueError("training dimension doesn't match")

        for i in range(len(train_doc_list)):
            self.initial_states_prob[train_doc_list[i][0]] += 1
            for j in range(len(train_doc_list[i])-1):
                self.trans_prob[train_doc_list[i][j]][train_doc_list[i][j+1]] += 1
                self.emission_prob[train_doc_list[i][j]][train_section_list[i][j]] += 1
            self.emission_prob[train_doc_list[i][-1]][train_section_list[i][-1]] += 1

        temp_sum_initial = sum(self.initial_states_prob)

        for i in range(len(self.trans_prob)):
            self.initial_states_prob[i] = float(self.initial_states_prob[i] / temp_sum_initial)
            temp_sum_trans = sum(self.trans_prob[i])
            for j in range(len(self.trans_prob[i])):
                self.trans_prob[i][j] = float(self.trans_prob[i][j] / temp_sum_trans)

        for i in range(len(self.emission_prob)):
            temp_sum_emission = sum(self.emission_prob[i])
            for j in range(len(self.emission_prob[i])):
                self.emission_prob[i][j] = float(self.emission_prob[i][j] / temp_sum_emission)

        self.save()

    def recover(self, observed_section_list: List[int]):
        """
        recover the hidden doc type list from observed list
        :param observed_section_list: list of sections indicated by integer index, in order
        :return:
        """
        DP = np.zeros((len(self.document_types), len(observed_section_list)))
        output_list = []
        for j in range(len(observed_section_list)):
            for i in range(len(self.document_types)):
                if j == 0:
                    DP[i][0] = math.log(self.initial_states_prob[i]) + \
                               math.log(self.emission_prob[i][observed_section_list[0]])
                else:
                    temp_list = []
                    for k in range(len(self.document_types)):
                        temp_list.append(DP[k][j-1] + math.log(self.trans_prob[k][i]) +
                                         math.log(self.emission_prob[i][observed_section_list[j]]))
                        DP[i][j] = max(temp_list)
            output_list.append(np.argmax(DP[:, j]))
        return output_list

    def save(self):
        d = {
            'initial': self.initial_states_prob,
            'trans': self.trans_prob,
            'emission': self.emission_prob
        }
        write_json(d, self.__get_output_file_name())

    def load(self):
        d = read_json(self.__get_output_file_name())

        self.initial_states_prob = d['initial']
        self.trans_prob = d['trans']
        self.emission_prob = d['emission']

    def __get_output_file_name(self) -> str:
        return os.path.join(RESULTS_PATH, self.model_file_name + '.json')
