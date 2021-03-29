import datetime

import tensorflow as tf
from keras import layers
import numpy as np
import typing

from logging_config import log_decorator


def load__data():
    return tf.keras.datasets.cifar10.load_data()


@log_decorator
class CNN:
    """
    Creating and Train CNN neural network with defined hyper-parameters
    """

    def __init__(self, hyper_parameters: typing.Dict, load_data=load__data, verbose=0):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data()
        self.hyper_parameters = hyper_parameters
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(self.x_train[0].shape))

        # Preprocessing Layers
        max_val = np.max(self.x_train.flatten)
        self.model.add(tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / max_val))
        self.model.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        self.model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2))

        # convolution layers (Feature extraction)
        cov_layers = self.hyper_parameters['convolution_layer']
        filters = self.hyper_parameters['pool_layers_filter']
        pool_layers = self.hyper_parameters['pool_layers_units']
        strides = self.hyper_parameters['pool_layers_strides']
        for layer, (cov, filt, pool, stride) in enumerate(zip(cov_layers, filters, pool_layers, strides)):
            if layer == 0:
                self.model.add(tf.keras.layers.Conv2D(cov, (filt, filt), strides=stride, activation='relu',
                                                      input_shape=self.input_shape))
            else:
                self.model.add(tf.keras.layers.Conv2D(cov, (filt, filt), strides=stride, activation='relu'))
            self.model.add(tf.keras.layers.MaxPool2D((pool, pool)))
        # Fully connected layers
        for ann_layer, dp_layer in zip(self.hyper_parameters['fully_connected_layers'],
                                       self.hyper_parameters['drop_layers']):
            self.model.add(tf.keras.layers.Dense(ann_layer, activation='relu'))
            if dp_layer > 0:
                self.model.add(tf.keras.layers.Dropout(dp_layer))
        # Output layer
        self.model.add(tf.keras.layers.Dense(self.output_size, activation='relu'))

        # Compile and Training Model
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        start_time = datetime.datetime.now()
        self.History = self.model.fit(self.x_train,
                                      self.y_train,
                                      epochs=self.hyper_parameters['epochs'],
                                      batch_size=self.hyper_parameters['batch_size'],
                                      validation_data=(self.x_test, self.y_train),
                                      verbose=verbose)
        self.Training_time = (datetime.datetime.now() - start_time).total_seconds() * 1000

    def __repr__(self):
        out_str = '-----------------------\nHyper-parameters:-\n'
        for hp, hp_val in self.hyper_parameters.items():
            out_str += f'{hp} : \t{hp_val}\n'
        out_str += 'Performance:-\n'
        acc_test, acc_train = self.accuracy
        out_str += f'Test  accuracy :\t{acc_test * 100:f00.0000}%\n'
        out_str += f'Train accuracy :\t{acc_train * 100:f00.0000}%\n'
        out_str += f'Train Time     :\t{self.Training_time} ms\n'
        out_str += '-----------------------'
        return out_str

    @property
    def accuracy(self):
        _, acc_train = self.model.evaluate(x=self.x_train, y=self.y_train)
        _, acc_test = self.model.evaluate(x=self.x_test, y=self.y_test)
        return acc_test, acc_train

    @property
    def input_shape(self):
        """
        The CNN Model input shape
        :return: The CNN Model input shape
        """
        return self.x_train[0].shape

    @property
    def output_size(self):
        """

        :return:
        """
        return len(np.unique(np.array(self.y_train).flatten()))

    @staticmethod
    def add_change_log(hp_dict, change_log):
        """
        Logging then changes that have been tuned to arrived to this set of hyper_parameters
        :param hp_dict: the hyper_parameters dictionary that hold the hyper_parameters set
        :param change_log: the change log text
        :return: The hyper_parameters dictionary with the change log updated
        """
        if 'changes' not in hp_dict.keys():
            hp_dict['changes'] = change_log + '\n'
        else:
            hp_dict['changes'] += change_log + '\n'
        return hp_dict

    # Model Changes
    @log_decorator
    def change_epoch(self, change):
        """
        Increase or decrease the number of epochs
        :param change: Amount of Epochs to increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with Epoch change updated
        """
        hp = self.hyper_parameters
        hp['epochs'] += np.int(change)
        hp['epochs'] = hp['epochs'] if hp['epochs'] > 1 else 1
        hp = self.add_change_log(hp, f'Epochs increased by {change}')
        return hp

    @log_decorator
    def change_batch_size(self, change):
        """
        Increase or decrease the batch size
        :param change: The batch size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with batch size change updated
        """
        hp = self.hyper_parameters
        hp['batch_size'] += np.int(change)
        hp['batch_size'] = hp['batch_size'] if hp['batch_size'] > 1 else 1
        hp = self.add_change_log(hp, f'Batch size increased by {change}')
        return hp

    @log_decorator
    def change_ann_layer_size(self, change):
        """
        Increase or decrease a randomly selected fully connected layer size
        :param change: The layer size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with layer size change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['fully_connected_layers'])
        layer_to_change = np.random.randint(num_of_layers)
        layer_size = hp['fully_connected_layers'][layer_to_change]
        layer_size += np.int(change)
        layer_size = layer_size if layer_size > self.output_size else self.output_size
        hp['fully_connected_layers'][layer_to_change] = layer_size
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} Fully connected layer size increased by {change}')
        return hp


def th(num: int) -> str:
    the_th = {1: 'st', 2: 'nd', 3: 'rd'}
    return f'{num}{the_th.get(num, "th")}'
