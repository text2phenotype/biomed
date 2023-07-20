from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers


# can also put this online part method under model class

def online_update(model_file, online_data):
    """
    this is to online update a model with new data
    :param model_file: the pre-trained model
    :param online_data: should be labeled x_train_np, y_train_np
    """
    # step 1, load model
    # step 2, feed this batch of new data to train
    # step 3, when compile can lower the learning rate, if we were to batch train, set a batch size smaller to the new data size
    # create a sequence from the click -> sliding window back and forward to create sequences ->

    model = load_model(model_file)

    # set the optimizer with different learning rate
    # define a new optimizer with smaller learning rate? what is the default learning rate for adam optimizer

    sgd = optimizers.SGD(lr=0.01)  # why not keep using Adam optimizer, what is the default learning rate for
    # need to set smaller learning rate, need to look up the default learning rate for initial training

    model.compile(optimizer=sgd, loss='categorical_crossentropy')  # why binary classification?

    x_train_np = online_data[0]

    y_train_np = online_data[1]

    model.fit(x_train_np,
              y_train_np,
              epoches=2,
              verbose=2,
              batch_size=128
              )

    # how to save model XXX with a new name?
    model.save(model_file[:-3] + '_v1' + '.h5', save_format="h5")
