import datetime
from keras import backend as K
import tensorflow as tf
from keras import layers
import numpy as np
import typing
from logging_config import log_decorator


def load__data():
    return tf.keras.datasets.cifar10.load_data()


def display_dict(d) -> str:
    out_str = ''
    for key, val in d.items():
        out_str += f'{key} : \t{val}\n'
    return out_str


def create_random_hyper_parameter():
    # initial number of layers is 2 (each of CNN and ANN layers)
    learning_rate_decay_function = [
        'constant',
        'PiecewiseConstantDecay',
        'exponential_decay',
        'InverseTimeDecay',
        'PolynomialDecay',
    ]

    hp = {
        'convolution_layer': np.random.randint(low=23, high=128, size=2),
        'pool_layers_filter': 2 ** np.random.randint(low=1, high=3, size=2),
        'pool_layers_units': np.random.randint(low=23, high=128, size=2),
        'pool_layers_strides': np.ones(2),
        'fully_connected_layers': np.random.randint(low=23, high=128, size=2),
        'drop_layers': np.zeros(2),  # no dropout layers initially
        'learning_rate': 0.01 * np.random.random(),
        'learning_rate_global_step': 100000,
        'learning_rate_decay_rate': 0.5 * np.round(np.random.random(), decimals=4),
        'learning_rate_decay_type': np.random.choice(learning_rate_decay_function),
        'epochs': np.random.randint(low=32, high=128),
        'batch_size': np.random.randint(low=16, high=128),
    }
    return hp


@log_decorator
class CNN:
    """
    Creating and Train CNN neural network with defined hyper-parameters
    """

    def __init__(self, hyper_parameters: typing.Dict, load_data=load__data, verbose=0):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data()
        self.hyper_parameters = hyper_parameters

        # Creating the model
        self.model = tf.keras.Sequential(name=self.hyper_parameters['name'])
        self.model.add(tf.keras.layers.Input(self.x_train[0].shape))

        # Preprocessing Layers
        max_val = np.max(self.x_train.flatten)
        self.model.add(layers.experimental.preprocessing.Rescaling(scale=1.0 / max_val))
        self.model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        self.model.add(layers.experimental.preprocessing.RandomRotation(0.2))

        # convolution layers (Feature extraction)
        cov_layers = self.hyper_parameters['convolution_layer']
        filters = self.hyper_parameters['pool_layers_filter']
        pool_layers = self.hyper_parameters['pool_layers_units']
        strides = self.hyper_parameters['pool_layers_strides']
        for layer, (cov, filt, pool, stride) in enumerate(zip(cov_layers, filters, pool_layers, strides)):
            if layer == 0:
                self.model.add(layers.Conv2D(cov, (filt, filt), strides=stride, activation='relu',
                                             input_shape=self.input_shape))
            else:
                self.model.add(layers.Conv2D(cov, (filt, filt), strides=stride, activation='relu'))
            self.model.add(layers.MaxPool2D((pool, pool)))
        # Fully connected layers
        for ann_layer, dp_layer in zip(self.hyper_parameters['fully_connected_layers'],
                                       self.hyper_parameters['drop_layers']):
            self.model.add(tf.keras.layers.Dense(ann_layer, activation='relu'))
            if dp_layer > 0:
                self.model.add(layers.Dropout(dp_layer))
        # Output layer
        self.model.add(tf.keras.layers.Dense(self.output_size, activation='relu'))

        # Compile and Training Model

        # select learning rate schedule
        learning_rate = self.hyperparameters['learning_rate']
        global_step = self.hyperparameters['learning_rate_global_step']
        decay_rate = self.hyperparameters['learning_rate_decay_rate']
        learning_rate_scheduler_function = {
            'Constant': tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[1000],
                values=[learning_rate, learning_rate]
            ),
            'PiecewiseConstantDecay': tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[global_step],
                values=[learning_rate, learning_rate - decay_rate]
            ),
            'exponential_decay': tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=global_step,
                decay_rate=decay_rate
            ),
            'InverseTimeDecay': tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=learning_rate,
                decay_steps=global_step,
                decay_rate=decay_rate
            ),
            'PolynomialDecay': tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=global_step,
                power=decay_rate
            ),
        }

        lr_schedule = learning_rate_scheduler_function[self.hyperparameters['learning_rate_decay_type']]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        start_time = datetime.datetime.now()
        self.History = self.model.fit(self.x_train,
                                      self.y_train,
                                      epochs=self.hyper_parameters['epochs'],
                                      batch_size=self.hyper_parameters['batch_size'],
                                      validation_data=(self.x_test, self.y_train),
                                      verbose=verbose)
        self.Training_time = (datetime.datetime.now() - start_time).total_seconds() * 1000  # in ms

    @classmethod
    def create_random_cnn_model(cls):
        return cls(create_random_hyper_parameter())

    def __repr__(self):
        out_str = '-----------------------\nHyper-parameters:-\n'
        out_str += display_dict(self.hyperparameters)
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
        Classification class count
        :return: Classification class count
        """
        return len(np.unique(np.array(self.y_train).flatten()))

    @property
    def trainable_parameters_count(self):
        """
        Get the trainable parameters count of the CNN model
        :return: Get the trainable parameters count of the CNN model
        """
        return int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))

    @staticmethod
    def add_change_log(hp_dict, change_log):
        """
        Logging then changes that have been tuned to arrived to this set of hyper_parameters
        :param hp_dict: the hyper_parameters dictionary that hold the hyper_parameters set
        :param change_log: the change log text
        :return: The hyper_parameters dictionary with the change log updated
        """
        if 'changes' not in hp_dict.keys():
            hp_dict['change_log'] = change_log + '\n'
        else:
            hp_dict['change_log'] += change_log + '\n'
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

    @log_decorator
    def change_cnn_layer_size(self, change):
        """
        Increase or decrease a randomly selected convolution layer size
        :param change: The layer size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with layer size change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['convolution_layer'])
        layer_to_change = np.random.randint(num_of_layers)
        layer_size = hp['convolution_layer'][layer_to_change]
        layer_size += np.int(change)
        layer_size = layer_size if layer_size > self.output_size else self.output_size
        hp['convolution_layer'][layer_to_change] = layer_size
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} convolution layer size increased by {change}')
        return hp

    @log_decorator
    def change_pool_layer_filter_size(self, change):
        """
        Increase or decrease a randomly selected pool layer filter size
        :param change: The layer size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with layer size change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['pool_layers_filter'])
        layer_to_change = np.random.randint(num_of_layers)
        layer_size = hp['pool_layers_filter'][layer_to_change]
        layer_size += np.int(change)
        layer_size = layer_size if layer_size > 2 else 2
        hp['pool_layers_filter'][layer_to_change] = layer_size
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} pool layer filter size increased by {change}')
        return hp

    @log_decorator
    def change_pool_layer_strides_size(self, change):
        """
        Increase or decrease a randomly selected pool layer filter size
        :param change: The layer size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with layer size change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['pool_layers_strides'])
        layer_to_change = np.random.randint(num_of_layers)
        layer_size = hp['pool_layers_strides'][layer_to_change]
        layer_size += np.int(change)
        layer_size = layer_size if layer_size > 1 else 1
        hp['pool_layers_strides'][layer_to_change] = layer_size
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} pool layer strides size increased by {change}')
        return hp

    @log_decorator
    def change_pool_layer_units_size(self, change):
        """
        Increase or decrease a randomly selected pool layer units size
        :param change: The layer size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with layer size change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['pool_layers_units'])
        layer_to_change = np.random.randint(num_of_layers)
        layer_size = hp['pool_layers_units'][layer_to_change]
        layer_size += np.int(change)
        layer_size = layer_size if layer_size > 1 else 1
        hp['pool_layers_units'][layer_to_change] = layer_size
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} pool layer units size increased by {change}')
        return hp

    @log_decorator
    def change_drop_layer(self, change):
        """
        Increase or decrease a randomly selected dropout layer drop factor
        :param change: The layer drop factor where to increase it (positive float)or decrease it (negative float)
        :return: The hyper_parameters dictionary with dropout layer change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['drop_layers'])
        layer_to_change = np.random.randint(num_of_layers)
        layer_size = hp['drop_layers'][layer_to_change]
        layer_size += np.int(change)
        layer_size = layer_size if layer_size > 0 else 0
        layer_size = layer_size if layer_size < 0.9 else 0.9
        hp['drop_layers'][layer_to_change] = layer_size
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} Dropout layer drop factor changed by {change}')
        return hp

    @log_decorator
    @property
    def change_learning_rate_decay_type(self):
        decay_type = ['Constant',
                      'PiecewiseConstantDecay',
                      'exponential_decay',
                      'InverseTimeDecay',
                      'PolynomialDecay']
        hp = self.hyper_parameters
        current_decay_type = hp['learning_rate_decay_type']
        decay_type.remove(current_decay_type)
        new_decay_type = np.random.choice(decay_type)
        hp['learning_rate_decay_type'] = new_decay_type
        if new_decay_type == 'Constant':
            decay_prarm = f'with learning rate {hp["learning_rate"]}'
        else:
            decay_rate = hp['learning_rate_decay_rate']
            step = hp['learning_rate_global_step']
            decay_rate = np.round(0.95 * np.random.random(), decimals=5)
            step = np.random.randint()
            hp['learning_rate_decay_rate'] = decay_rate
            hp['learning_rate_global_step'] = step
            decay_prarm = f'with learning rate {hp["learning_rate"]}, Global step {step} and decay rate {decay_rate}'
        hp = self.add_change_log(hp, f'learning rate decay type changed to {new_decay_type} {decay_prarm}')
        return hp


def th(num: int) -> str:
    the_th = {1: 'st', 2: 'nd', 3: 'rd'}
    return f'{num}{the_th.get(num, "th")}'
