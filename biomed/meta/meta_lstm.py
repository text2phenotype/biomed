import numpy as np

from tensorflow.keras.layers import Bidirectional, Dense, TimeDistributed, LSTM
from tensorflow.keras import Sequential
from scipy.stats import entropy

from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.common import common
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.common.log import operations_logger

from biomed.models.model_base import ModelBase
from biomed.constants.model_enums import ModelType


class MetaLSTM(ModelBase):
    def __init__(self):
        super().__init__(model_type=ModelType.meta)

    def train(self):
        """
        this function is train a ensembler LSTM
        """
        training_file_list = self.data_source.get_matched_annotated_files(self.label_feature)

        train_x, train_y, _ = self.prepare(training_file_list)

        feature_dim = len(train_x[0])
        num_classes = len(train_y[0])

        x_train_np, y_train_np = self.reshape_labeled(train_x, train_y)

        model = Sequential()

        model.add(Bidirectional(LSTM(self.job_metadata.hidden_dim, return_sequences=True),
                                input_shape=(self.model_metadata.window_size, feature_dim), merge_mode='concat'))

        model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

        model.fit(x_train_np,
                  y_train_np,
                  epochs=self.job_metadata.epochs,
                  verbose=2,
                  batch_size=self.job_metadata.batch_size)

        return self.save(model)

    def prepare(self, file_list):
        """
        this function overwrite the base class, takes the training files and transfer them into trainable data
        :return: 3D matrix and label vector
        """
        model_type_enum = self.model_metadata.model_type
        temp_training_matrix = None
        training_matrix = []  # of dimension [num_tokens, total num of each model's output class dimension]
        label_class = []
        token_list = []
        first_file = True
        model_index = MODEL_TYPE_2_CONSTANTS[self.model_type].label_class
        feature_service_client = FeatureServiceClient()

        for file in file_list:
            tokens = MachineAnnotation(json_dict_input=common.read_json(file))
            vectors = feature_service_client.vectorize(tokens)
            first_model = True
            for model_file in self.model_metadata.base_classifier_list:
                model = ModelBase(file_name=model_file, model_type=self.model_type)
                if first_model:
                    temp_training_matrix = model.predict(tokens, vectors)['weighted_encoding']
                    first_model = False
                else:
                    temp_train_x = model.predict(tokens, vectors)['weighted_encoding']
                    temp_training_matrix = np.append(temp_training_matrix, temp_train_x, axis=1)
            if first_file:
                training_matrix = temp_training_matrix
                first_file = False
            else:
                training_matrix = np.append(training_matrix, temp_training_matrix, axis=0)
                del temp_training_matrix
            for token in tokens:
                token_list.append(token['token'])

                label_vector = [0] * len(model_index)
                if model_type_enum.name + '_label' not in token:
                    label_vector[0] = 1
                else:
                    label_vector[model_index[token[model_type_enum.name + '_label']].value] = 1
                label_class.append(label_vector)
        return training_matrix, label_class, token_list

    def predict(self, tokens, vectors):
        """
        predict a result of some model type using the meta-classifier.
        """
        model_index = MODEL_TYPE_2_CONSTANTS[self.model_type].label_class
        first_model = True
        for model_file in self.model_metadata.base_classifier_list:
            model = ModelBase(file_name=model_file, model_type=self.model_type)
            if first_model:
                feature_matrix_x = model.predict(tokens, vectors)['weighted_encoding']
                first_model = False
            else:
                temp_feature_matrix = model.predict(tokens, vectors)['weighted_encoding']
                feature_matrix_x = np.append(feature_matrix_x, temp_feature_matrix, axis=1)
                del temp_feature_matrix
        # TODO: test, where this model file was gotten is it obtained from the meta package?
        x_test_np = self.reshape_unlabeled(feature_matrix_x)
        model = self.get_cached_model()
        y_pred_class = model.predict_classes(x_test_np, self.job_metadata.batch_size)
        operations_logger.debug(f"Predicted Y shape is: {y_pred_class.shape}")
        if y_pred_class.shape[0] == 1:
            y_voted_np = y_pred_class[0]
        else:
            y_voted_np = self.vote_majority(y_pred_class, len(tokens['token']))
        if y_voted_np.shape[0] != len(tokens) and y_pred_class.shape[0] != 1:
            operations_logger.debug("Prediction dimension and data dimension don't match!")
        output = list()
        for i in range(len(tokens)):
            output.append({model_index(y_voted_np[i]).name: (tokens[i]['token'], tokens[i]['range'])})
        y_pred_prob = model.predict(x_test_np, self.job_metadata.batch_size)
        if y_pred_prob.shape[0] == 1:
            y_voted_weight_np = y_pred_prob[0]
        else:
            # should be of dimension [number of tokens, num_classes]
            y_voted_weight_np = self.vote_with_weight(y_pred_prob, len(tokens['token']))
        num_classes = y_pred_prob.shape[1]
        average_probability = float(1/num_classes)
        y_voted_weight_category = np.argmax(y_voted_weight_np, axis=1)
        y_voted_weight_prob = np.amax(y_voted_weight_np, axis=1)
        output_with_prob = list()
        output_with_prob_print = list()
        total_entropy = 0
        uncertain_count = 0
        uncertain_tokens = {}
        for i in range(len(tokens)):
            output_with_prob.append({self.model_metadata.model_type.name: model_index(y_voted_weight_category[i]).name,
                                     'token': tokens[i]['token'],
                                     'range': tokens[i]['range'],
                                     'prob': round(float(y_voted_weight_prob[i]), 3)})
            output_with_prob_print.append({
                model_index(y_voted_weight_category[i]).name: (round(float(y_voted_weight_prob[i]), 3),
                                                               tokens[i]['token'],
                                                               tokens[i]['range'])
            })
            if self.job_metadata.return_uncertainty:
                total_entropy += entropy(y_voted_weight_np[i])
            if not self.job_metadata.narrow_band:
                if float(y_voted_weight_prob[i]) < (
                        average_probability + self.job_metadata.narrow_band):
                    uncertain_count += 1
                    if model_index(y_voted_weight_category[i]).name not in uncertain_tokens:
                        uncertain_tokens[model_index(y_voted_weight_category[i]).name] = list()
                    uncertain_tokens[model_index(y_voted_weight_category[i]).name].append((
                        round(float(y_voted_weight_prob[i]), 3),
                        tokens[i]['token'],
                        tokens[i]['range']))
        return {'majority': output,
                'weighted': output_with_prob_print,
                'weighted_encoding': y_voted_weight_np,
                'average_entropy': round(float(total_entropy / y_voted_weight_np.shape[0]), 3),
                'uncertain_count': uncertain_count,
                'narrow_band_ratio': round(float(uncertain_count / len(tokens)), 3),
                'uncertain_tokens': uncertain_tokens,
                'narrow_band_width': self.job_metadata.narrow_band,
                'tokens': [(tokens[i]['token'], tokens[i]['range']) for i in range(len(tokens))]}
