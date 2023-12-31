from math import ceil
import numpy
from typing import List

from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    LSTM,
    Masking,
    RNN,
    SimpleRNN,
    SimpleRNNCell,
)
from tensorflow.keras.models import Sequential

from text2phenotype.common import aspect_samples
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.ccda.section import Aspect
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType


class AspectModel:
    def __init__(self):
        self.data_source = None
        self.batch_size = None
        self.hidden_dim = None
        self.epochs = None

    def train(self) -> str:
        """
        train a bi-directional LSTM model with a dense softmax layer for 12 aspects, ignoring the
        sequential dependency for aspect with document for now, focus on the sequential dependency
        of tokens within the aspect
        :return: trained model
        """
        training_files, total_token_count = self.data_source.get_matched_files_and_token_size('aspect')

        x_train, y_train = self.prepare([training_files[0]])
        feature_dim = len(x_train[0])
        num_classes = len(y_train[0])

        # calculate steps per epoch
        steps_per_epoch = ceil(total_token_count / self.batch_size)

        operations_logger.debug(f'Feature Dimension is: {feature_dim}')
        operations_logger.debug(f'Number of Classes is: {num_classes}')
        operations_logger.debug(f'Number of Steps Per Epoch: {steps_per_epoch}')

        # TODO: try pad sequence to pad the sequence length to be the maximum sequence length with 0s
        model = Sequential()  # initialize the sequential model
        # maybe make more sense to use a one directional LSTM than bi-directional
        model.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=False),
                                input_shape=(None, feature_dim),
                                merge_mode='concat'))
        # when initialize the LSTM neuron, set return_sequence to be False, as we just need one output
        # from the middle layer.
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adagrad')
        if len(self.data_source.validation_dirs) < 1:
            model.fit_generator(self.generator(training_files),
                                epochs=self.epochs, steps_per_epoch=steps_per_epoch, verbose=2)
        else:
            validation_files = self.data_source.get_validation_files('ModelType.demographic')
            model.fit_generator(self.generator(training_files),
                                epochs=self.epochs, steps_per_epoch=steps_per_epoch,
                                verbose=2, validation_data=self.generator(validation_files),
                                validation_steps=ceil(
                                    self.data_source.get_token_size(validation_files) / self.batch_size))

        return self.save(model)

    @staticmethod
    def save(self):
        return ''  # TODO(mjp): I don't think this works as expected

    @staticmethod
    def generator(self):
        return None

    def test(self) -> str:
        pass

    def predict(self, tokens: list):
        pass

    def prepare(self, file_paths: List[str]):
        """
        prepare training matrix and label matrix for rnn with the training file
        :param file_paths: files to load training data
        :return: numpy array of training examples and labels
        """
        client = FeatureServiceClient()
        feature_matrix = []
        label_matrix = []
        i = 0
        with open(file_paths[0], 'r') as f_r:
            for line in f_r:  # for each example, compose a training vector and training label vector
                # test first 100 lines
                line_list = line.split('|')
                if len(line_list) < 2:
                    continue
                _text = str(line_list[1].strip())
                if _text == 'None.' or not _text:
                    continue
                if i >= 10000:
                    break
                i += 1
                if i % 100 == 0:
                    operations_logger.info("%s examples processed" % str(i))
                text_matrix = []
                label_vector = [0] * len(Aspect.get_active_aspects())
                _class = int(line.split('|')[0].strip())  # get the class label
                features = [feature_type for feature_type in FeatureType if feature_type is not FeatureType.aspect]
                _tokens = client.annotate_text(_text, features)  # annotate text without assigning aspect
                # maybe make more sense to change the featureset not to contain the
                # aspect feature or the tf feature
                vectorized_tokens = client.vectorize_from_annotations(_tokens,
                                                                      self.binary_classifier,
                                                                      self.features)
                for token in vectorized_tokens:
                    token_featureset = []
                    for dimension in sorted(token.keys()):
                        if dimension not in self.EXCLUDED_LABELS:
                            token_featureset += token[dimension]
                    # token_featureset is of dimension 1 * feature dim (199)
                    text_matrix.append(token_featureset)
                label_vector[_class] = 1
                # TODO: may do some transformation/or scaling for text_matrix down to single vector
                # TODO: e.g. convolution: according to this post:
                # TODO: http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
                # TODO: or normalize it across the column to be a single dimension vector, e.g. sum(X) or sum(X)/N
                # text_matrix dimension is about [#tokens * #feature dim] note: # of tokens may change,
                # can use max_len or pad sequence to normalize them to be the same length
                # dimension [#sequences(examples), #tokens in each example, #featuredim]
                feature_matrix.append(text_matrix)
                # dimension [#seq, #classes]
                label_matrix.append(label_vector)
        # print ("Feature Matrix Dimension is:")
        # print (feature_matrix.shape[0], feature_matrix.shape[1], feature_matrix.shape[2])
        # print ("Label Matrix Dimension is:")
        # print (label_matrix.shape[0], label_matrix.shape[1])
        return feature_matrix, label_matrix

    def return_prediction(self, x_input, test_token):
        pass

    def train_forward_lstm_from_bsv(self, file_bsv):
        """
        initialize an one directional RNN model for aspect labeler, denote early exit before padded vectors
        with input_length containing lengths of each sequence. or input
        :return: one-directional rnn with input length
        """
        file_path = aspect_samples.train_path(file_bsv)
        # train_x dimension [#seq, #token, #featuredim] train_y [#seq, #class]
        train_x, train_y = self.prepare([file_path])
        # TODO: before we feed into model.fit for training, should padding the remaining sequences with 0 vectors to be
        # TODO: the same length to use keras.preprocess.pad_sequence, which currently just takes in a list of lists, we
        # TODO: can reduce the 3D tensor dimension down to be 2D, such as down to a matrix of feaure_dim and seq_length,
        # TODO: and fill that in with zeros
        train_x_np, max_len, input_length = self.pad_sequences(train_x)
        # have to reshape train_x to be 3D so that it can be consumed by model.fit
        train_y_np = numpy.array(train_y)
        train_x_np = numpy.array(train_x)
        num_classes = train_y_np.shape[1]
        feature_dim = train_x_np.shape[2]
        # TODO: try pad sequence to pad the sequence length to be the maximum sequence length with 0s
        model = Sequential()  # initialize the sequential model
        # maybe make more sense to use a one directional LSTM than bi-directional
        model.add(LSTM(self.hidden_dim, return_sequences=False, input_shape=(None, feature_dim)))
        # when initialize the LSTM neuron, set return_sequence to be False, as we just need one output
        # from the middle layer.
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adagrad')
        model.fit_generator(train_x_np,
                            train_y_np,
                            epochs=self.epochs,
                            verbose=1,
                            batch_size=self.batch_size)
        return self.save(model)

    def train_forward_rnn_from_bsv(self, file_bsv):
        """
        Train a simple forward simple recurrent neural net, this might not be as good as lstm,
        just to explore early exit and masking option here
        """
        file_path = aspect_samples.train_path(file_bsv)
        # train_x dimension [#seq, #token, #featuredim] train_y [#seq, #class]
        train_x, train_y = self.prepare([file_path])
        # TODO: before we feed into model.fit for training, should padding the remaining sequences with 0 vectors to be
        # TODO: the same length to use keras.preprocess.pad_sequence, which currently just takes in a list of lists, we
        # TODO: can reduce the 3D tensor dimension down to be 2D, such as down to a matrix of feaure_dim and seq_length,
        # TODO: and fill that in with zeros
        train_x_np, max_len, input_length = self.pad_sequences(train_x)
        # have to reshape train_x to be 3D so that it can be consumed by model.fit
        train_y_np = numpy.array(train_y)
        train_x_np = numpy.array(train_x)
        num_classes = train_y_np.shape[1]
        feature_dim = train_x_np.shape[2]
        # TODO: try pad sequence to pad the sequence length to be the maximum sequence length with 0s
        model = Sequential()  # initialize the sequential model
        # maybe make more sense to use a one directional LSTM than bi-directional
        # TODO: do we have the option to have a input_length here see the doc:
        model.add(RNN(SimpleRNN(self.hidden_dim),
                      return_sequences=False,
                      input_shape=(None, feature_dim)))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adagrad')
        model.fit(train_x_np,
                  train_y_np,
                  epochs=self.epochs,
                  verbose=1,
                  batch_size=self.batch_size)
        return self.save(model)

    def train_forward_lstm_with_mask(self, file_bsv):
        """
        train a forward LSTM network with dense layer with masking the 0 input, i.e., skip the 0 vectors
        """
        file_path = aspect_samples.train_path(file_bsv)
        # train_x dimension [#seq, #token, #featuredim] train_y [#seq, #class]
        train_x, train_y = self.prepare([file_path])
        # TODO: before we feed into model.fit for training, should padding the remaining sequences with 0 vectors to be
        # TODO: the same length to use keras.preprocess.pad_sequence, which currently just takes in a list of lists, we
        # TODO: can reduce the 3D tensor dimension down to be 2D, such as down to a matrix of feaure_dim and seq_length,
        # TODO: and fill that in with zeros
        train_x_np, max_len, input_length = self.pad_sequences(train_x)
        # have to reshape train_x to be 3D so that it can be consumed by model.fit
        train_y_np = numpy.array(train_y)
        # train_x_np = np.array(train_x)
        num_classes = train_y_np.shape[1]
        time_steps = train_x_np.shape[1]
        feature_dim = train_x_np.shape[2]
        # TODO: try pad sequence to pad the sequence length to be the maximum sequence length with 0s
        model = Sequential()  # initialize the sequential model
        model.add(Masking(mask_value=0, input_shape=(time_steps, feature_dim)))
        # maybe make more sense to use a one directional LSTM than bi-directional
        # TODO: do we have the option to have a input_length here see the doc:
        model.add(LSTM(self.hidden_dim, return_sequences=False))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adagrad')
        model.fit(train_x_np,
                  train_y_np,
                  epochs=self.epochs,
                  verbose=1,
                  batch_size=self.batch_size)
        return self.save(model)

    def train_forward_rnn_with_mask(self, file_bsv):
        """
        train a forward LSTM network with dense layer with masking the 0 input, i.e., skip the 0 vectors
        """
        file_path = aspect_samples.train_path(file_bsv)
        # train_x dimension [#seq, #token, #featuredim] train_y [#seq, #class]
        train_x, train_y = self.prepare([file_path])
        # TODO: before we feed into model.fit for training, should padding the remaining sequences with 0 vectors to be
        # TODO: the same length to use keras.preprocess.pad_sequence, which currently just takes in a list of lists, we
        # TODO: can reduce the 3D tensor dimension down to be 2D, such as down to a matrix of feaure_dim and seq_length,
        # TODO: and fill that in with zeros
        train_x_np, max_len, input_length = self.pad_sequences(train_x)
        # have to reshape train_x to be 3D so that it can be consumed by model.fit
        train_y_np = numpy.array(train_y)
        train_x_np = numpy.array(train_x)
        num_classes = train_y_np.shape[1]
        time_steps = train_x_np.shape[1]
        feature_dim = train_x_np.shape[2]
        # TODO: try pad sequence to pad the sequence length to be the maximum sequence length with 0s
        model = Sequential()  # initialize the sequential model
        model.add(Masking(mask_value=0, input_shape=(time_steps, feature_dim)))
        # maybe make more sense to use a one directional LSTM than bi-directional
        # TODO: do we have the option to have a input_length here see the doc:
        model.add(RNN(SimpleRNNCell(self.hidden_dim), return_sequences=False))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adagrad')
        model.fit(train_x_np,
                  train_y_np,
                  epochs=self.epochs,
                  verbose=1,
                  batch_size=self.batch_size)
        return self.save(model)

    @staticmethod
    def pad_sequences(train_x):
        # TODO: maybe can use pad_sequences and list slicing and indexing to speed up
        """
        reshape the training matrix train_x(list of lists of lists) from dimension [#seq, #token(vary), #featuredim]
        to be of dimension [#seq, #max_token, #featuredim]
        :param train_x: training feature matrix defined
        :return:
        """
        max_length = 0
        input_length = []  # keep input sequence lengths of the input text
        feature_dim = len(train_x[0][0])
        for i in range(len(train_x)):
            input_length.append(len(train_x[i]))
            if len(train_x[i]) > max_length:
                max_length = len(train_x[i])
        for i in range(len(train_x)):
            if len(train_x[i]) < max_length:
                for j in range(max_length - len(train_x[i])):
                    train_x[i].append([0] * feature_dim)
        train_x = numpy.array(train_x)
        return train_x, max_length, input_length
