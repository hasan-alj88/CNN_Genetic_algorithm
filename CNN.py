import tensorflow as tf
from keras import layers
import typing


def load_data():
    return tf.keras.datasets.cifar10.load_data()


class CNN:
    """:cvar
    Creating and Train CNN neural network with defined hyper-parameters
    """

    def __init__(self, hyper_parameters: typing.Dict, Load_data=load_data):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = Load_data()
        self.hyper_parameters = hyper_parameters
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(self.x_train[0].shape))

        # convolution layers (Feature extraction)
        cov_layers = self.hyper_parameters['Convolution_layer']
        filters = self.hyper_parameters['Pool_layers_filter']
        pool_layers = self.hyper_parameters['Pool_layers_units']
        strides = self.hyper_parameters['Pool_layers_strides']
        for layer, (cov, filt, pool, stride) in enumerate(zip(cov_layers, filters, pool_layers, strides)):
            if layer == 0:
                self.model.add(tf.keras.layers.Conv2D(cov, (filt, filt), strides=stride, activation='relu',
                                                      input_shape=self.input_shape))
            else:
                self.model.add(tf.keras.layers.Conv2D(cov, (filt, filt), strides=stride, activation='relu'))
            self.model.add(tf.keras.layers.MaxPool2D((pool, pool)))
        # Fully connected layers
        for ann_layer, dp_layer in zip(self.hyper_parameters['Fully_connected_layers'],
                                       self.hyper_parameters['Drop_layers']):
            self.model.add(tf.keras.layers.Dense(ann_layer, activation='relu'))
            if dp_layer > 0:
                self.model.add(tf.keras.layers.Dropout(dp_layer))

    @property
    def input_shape(self):
        return self.x_train[0].shape
