import datetime
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, List
from logging_config import log_decorator
from tensorflow_datasets import load
import pprint as pp
from icecream import ic
from functools import reduce


# @TECHREPORT{Krizhevsky09learningmultiple,
#     author = {Alex Krizhevsky},
#     title = {Learning multiple layers of features from tiny images},
#     institution = {},
#     year = {2009}
# }


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def load_data(batch_size):
    train_ds = load('cifar100', split='train', shuffle_files=True, as_supervised=True)
    test_ds = load('cifar100', split='test', shuffle_files=True, as_supervised=True)
    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    ic(train_ds)
    ic(test_ds)
    input_size = (32, 32, 3)  # including batch dimension
    output_size = 100
    return train_ds, test_ds, input_size, output_size


def zero_padding(i, s, f):
    return (s * (i // s) - i + f - s) // 2


def conv_output(i, s, f):
    return ((i - f + 2 * zero_padding(i, s, f)) // s) + 1


def generate_random_convolution_parameters(input_size: int = 32, number_of_layers: int = 3) -> Tuple[int, int, int]:
    i = np.random.randint(low=input_size // 2, high=2 * input_size)
    for _ in range(number_of_layers):
        # strides
        s = np.random.randint(low=1, high=np.ceil(np.log2(i)))
        # filters
        f = np.random.randint(low=2, high=6)
        yield i, f, s
        i = conv_output(i, s, f)


def create_random_hyper_parameter(input_size: int = 32, output_size: int = 100) -> Dict:
    # initial number of layers is 2 (each of CNN and ANN layers)
    learning_rate_decay_function = [
        'Constant',
        'PiecewiseConstantDecay',
        'exponential_decay',
        'InverseTimeDecay',
        'PolynomialDecay',
    ]

    conv_gen = generate_random_convolution_parameters(input_size, 2)
    conv_units, conv_filters, conv_strides = zip(*[_ for _ in conv_gen])

    hp = {
        'convolution_layer': conv_units,
        'convolution_layers_filter': conv_filters,
        'convolution_layers_strides': conv_strides,
        'fully_connected_layers': np.random.randint(low=output_size, high=256, size=3),
        'drop_layers': np.zeros(2),  # no dropout layers initially
        'learning_rate': np.round(0.01 * np.random.random(), decimals=6),
        'learning_rate_function_params': [np.round(np.random.random(), decimals=6), 1000],
        'learning_rate_decay_type': np.random.choice(learning_rate_decay_function),
        'epochs': np.random.randint(low=32, high=128),
        'batch_size': np.random.randint(low=8, high=100),
    }
    return hp


def th(num: int) -> str:
    the_th = {1: 'st', 2: 'nd', 3: 'rd'}
    return f'{num}{the_th.get(num, "th")}'


@log_decorator
class AnyDice:
    def __init__(self, probabilities: List[float]):
        if not np.isclose(np.sum([0, probabilities]), 1.0):
            raise ValueError('sum of probabilities must equal 1.0')
        probabilities.insert(0, 0)
        self.boundaries = np.add.accumulate(probabilities)

    @log_decorator
    @property
    def roll(self) -> int:
        the_roll = np.random.random()
        for interval, (lower, upper) in enumerate(zip(self.boundaries[:-1], self.boundaries[1:])):
            if lower <= the_roll <= upper:
                return interval
        else:
            return len(self.boundaries) - 1


class CNN:
    """
    Creating and Train CNN neural network with defined hyper-parameters
    """

    def __init__(self, hyper_parameters: Dict, verbose=0):
        self.ds_train, self.ds_test, self.input_shape, self.output_size = load_data(hyper_parameters['batch_size'])
        self.hyper_parameters = hyper_parameters

        # Creating the model
        self.model = tf.keras.Sequential(name=self.hyper_parameters['name'] + '_model')
        self.model.add(tf.keras.layers.InputLayer(
            input_shape=self.input_shape,
            name=self.hyper_parameters['name'] + '_InputLayer',
            # batch_size=self.hyper_parameters['batch_size']
        ))

        # convolution layers
        cov_layers = self.hyper_parameters['convolution_layer']
        filters = self.hyper_parameters['convolution_layers_filter']
        strides = self.hyper_parameters['convolution_layers_strides']
        for layer, (cov, filt, stride) in enumerate(zip(cov_layers, filters, strides)):
            if layer == 0:
                self.model.add(tf.keras.layers.Conv2D(cov,
                                                      (filt, filt),
                                                      strides=stride,
                                                      activation='relu',
                                                      padding='same',
                                                      input_shape=self.input_shape,
                                                      name=self.hyper_parameters['name'] + f'_conv{layer}'))
            else:
                self.model.add(tf.keras.layers.Conv2D(cov,
                                                      (filt, filt),
                                                      strides=stride,
                                                      activation='relu',
                                                      padding='same',
                                                      name=self.hyper_parameters['name'] + f'_conv{layer}'))
        # Fully connected layers
        self.model.add(tf.keras.layers.Flatten())
        for layer, (ann_layer, dp_layer) in enumerate(zip(
                self.hyper_parameters['fully_connected_layers'],
                self.hyper_parameters['drop_layers'])):
            self.model.add(tf.keras.layers.Dense(ann_layer,
                                                 activation='relu',
                                                 name=self.hyper_parameters['name'] + f'_FCL{layer}'))
            if dp_layer > 0:
                self.model.add(tf.keras.layers.Dropout(dp_layer, name=self.hyper_parameters['name'] + f'_Drop{layer}'))
        # Output layer
        self.model.add(tf.keras.layers.Dense(self.output_size,
                                             activation='softmax',
                                             name=self.hyper_parameters['name'] + f'_SoftMax'))
        self.model.summary()
        # Compile and Training Model
        for layer in self.model.layers:
            print(f'{layer.name}|{layer.input_shape} -> {layer.output_shape}')
        # select learning rate schedule
        learning_rate = self.hyper_parameters['learning_rate']
        decay_rate, global_step = self.hyper_parameters['learning_rate_function_params']

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

        lr_schedule = learning_rate_scheduler_function[self.hyper_parameters['learning_rate_decay_type']]
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        start_time = datetime.datetime.now()
        self.History = self.model.fit(self.ds_train,
                                      epochs=self.hyper_parameters['epochs'],
                                      validation_data=self.ds_test,
                                      verbose=verbose)
        self.Training_time = (datetime.datetime.now() - start_time).total_seconds() * 1000  # in ms

    @classmethod
    def create_random_cnn_model(cls, input_size: int):
        return cls(create_random_hyper_parameter(input_size))

    def __repr__(self):
        out_str = '-----------------------\nHyper-parameters:-\n'
        out_str += pp.pformat(self.hyper_parameters)
        out_str += '\nPerformance:-\n'
        acc_test, acc_train = self.accuracy
        out_str += f'Test  accuracy :\t{acc_test * 100:3.6f}%\n'
        out_str += f'Train accuracy :\t{acc_train * 100:3.6f}%\n'
        out_str += f'Train Time     :\t{self.Training_time:0.6f} ms\n'
        out_str += '-----------------------'
        return out_str

    @property
    def accuracy(self):
        _, acc_train = self.model.evaluate(self.ds_train)
        _, acc_test = self.model.evaluate(self.ds_test)
        return acc_test, acc_train

    @property
    def trainable_parameters_count(self):
        """
        Get the trainable parameters count of the CNN model
        :return: Get the trainable parameters count of the CNN model
        """
        return int(np.sum([tf.keras.backend.count_params(p) for p in set(self.model.trainable_weights)]))

    @staticmethod
    def add_change_log(hp_dict, change_log):
        """
        Logging then changes that have been tuned to arrived to this set of hyper_parameters
        :param hp_dict: the hyper_parameters dictionary that hold the hyper_parameters set
        :param change_log: the change log text
        :return: The hyper_parameters dictionary with the change log updated
        """
        if 'change_log' not in hp_dict.keys():
            hp_dict['change_log'] = change_log + '\n'
        else:
            hp_dict['change_log'] += change_log + '\n'
        return hp_dict

    def layer_output_size(self, layer):
        if layer < 0:
            return self.input_shape
        return self.model.layers[layer].output_shape[1]

    # Model Changes
    # Those functions will produce Hyper-parameter dictionary to be created with the changes
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
        hp = self.add_change_log(hp, f"Epochs changed by {change} where currently is {hp['epochs']}")
        return hp

    @log_decorator
    def change_batch_size(self, change: int):
        """
        Increase or decrease the batch size
        :param change: The batch size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with batch size change updated
        """
        hp = self.hyper_parameters
        hp['batch_size'] += np.int(change)
        hp['batch_size'] = hp['batch_size'] if hp['batch_size'] > 1 else 1
        hp = self.add_change_log(hp, f"Batch size is changed by {change} where its currently {hp['batch_size']}")
        return hp

    @log_decorator
    def change_learning_rate(self, change):
        hp = self.hyper_parameters
        hp['learning_rate'] += change
        self.add_change_log(hp, f"learning rate has been changed into {hp['learning_rate']}")
        return hp

    @log_decorator
    def change_add_ann_layer(self, change: int) -> Dict:
        hp = self.hyper_parameters
        ann_layers = np.array(hp['fully_connected_layers'])
        layer_placement = np.random.randint(low=0, high=len(ann_layers))
        hp['fully_connected_layers'] = np.insert(arr=ann_layers, obj=layer_placement, values=change)
        hp['drop_layers'] = np.insert(arr=hp['drop_layers'], obj=layer_placement, values=0)
        self.add_change_log(hp, f'Fully connected layer of size {change} has been added in {th(layer_placement)}' +
                            ' layer after flatten layer')
        return hp

    @log_decorator
    @property
    def change_add_cnn_layer(self) -> Dict:
        hp = self.hyper_parameters
        cnn_strides = np.array(hp['convolution_layers_strides'])
        cnn_filter = np.array(hp['convolution_layers_filter'])
        conv = np.array(hp['convolution_layer'])
        layer_placement = np.random.randint(low=0, high=len(conv))
        f = np.random.randint(low=2, high=8)
        i_m1 = self.input_shape[0] if layer_placement == 0 else conv[layer_placement - 1]
        i = conv_output(conv[i_m1], cnn_strides[layer_placement], cnn_filter[layer_placement])
        c = np.random.randint(low=i // 2 + 1, high=i * 2)
        hp['convolution_layers_strides'] = np.insert(arr=cnn_strides, obj=layer_placement, values=1)
        hp['convolution_layers_filter'] = np.insert(arr=cnn_filter, obj=layer_placement, values=f)
        hp['convolution_layer'] = np.insert(arr=conv, obj=layer_placement, values=c)
        self.add_change_log(hp,
                            f'CNN layer of kernel size {c} and filter ({f}x{f}) has been added in {th(layer_placement)}')
        return hp

    @log_decorator
    def change_ann_layer_size(self, change: int):
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
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} Fully connected layer size changed by {change}')
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
    def change_conv_layer_strides_size(self, change):
        """
        Increase or decrease a randomly selected convolution layer filter size
        :param change: The layer size increase (positive int)or decrease (negative int)
        :return: The hyper_parameters dictionary with layer size change updated
        """
        hp = self.hyper_parameters
        num_of_layers = len(hp['convolution_layers_strides'])
        layer_to_change = np.random.randint(num_of_layers)
        strides = hp['convolution_layers_strides'][layer_to_change]
        strides += np.int(change)
        i = self.layer_output_size(layer_to_change - 1)
        f = hp['convolution_layers_filter'][layer_to_change]
        output_size_after_change = conv_output(i, strides, f)
        strides = strides if output_size_after_change > 1 else 1
        hp['convolution_layers_strides'][layer_to_change] = strides
        hp = self.add_change_log(hp, f'The {th(layer_to_change)} convolution layer strides size increased by {change}')
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
    def change_learning_rate_decay_type(self) -> Dict:
        decay_type = ['Constant',
                      'PiecewiseConstantDecay',
                      'exponential_decay',
                      'InverseTimeDecay',
                      'PolynomialDecay']
        hp = self.hyper_parameters
        # Selecting new decay type
        current_decay_type = hp['learning_rate_decay_type']
        decay_type.remove(current_decay_type)
        new_decay_type = np.random.choice(decay_type)
        hp['learning_rate_decay_type'] = new_decay_type
        if new_decay_type == 'Constant':
            decay_prarm = f'with learning rate {hp["learning_rate"]}'
        else:
            decay_rate, step = hp['learning_rate_function_params']
            decay_rate = np.round(0.01 * np.random.random(), decimals=5)
            step = np.random.randint(low=10, high=1e4)
            hp['learning_rate_function_params'] = [decay_rate, step]
            decay_prarm = f'with learning rate {hp["learning_rate"]}, Global step {step} and decay rate {decay_rate}'
        hp = self.add_change_log(hp, f'learning rate decay type changed to {new_decay_type} {decay_prarm}')
        return hp

    @log_decorator
    @property
    def change_for_over_fit(self) -> Dict:
        prob = np.array([1, 1, 1])
        prob = prob / np.sum(prob)
        change_selection = AnyDice(prob).roll
        if change_selection == 0:
            return self.change_drop_layer(np.random.random() * 0.1)
        elif change_selection == 1:
            return self.change_ann_layer_size(-1)
        elif change_selection == 2:
            return self.change_learning_rate_decay_type

    @log_decorator
    @property
    def change_for_slow_training_time(self) -> Dict:
        prob = np.array([3, 2, 1])
        prob = prob / np.sum(prob)
        change_selection = AnyDice(prob).roll
        if change_selection == 0:
            return self.change_batch_size(np.random.randint(low=-16, high=16))
        elif change_selection == 1:
            return self.change_epoch(np.random.randint(low=-16, high=-4))
        else:
            return self.change_learning_rate_decay_type

    @log_decorator
    @property
    def change_for_under_fitting(self) -> Dict:
        prob = np.array([4, 3, 4, 3, 5, 5, 2, 1])
        prob = prob / np.sum(prob)
        change_selection = AnyDice(prob).roll
        if change_selection == 0:
            return self.change_ann_layer_size(np.random.randint(low=1, high=16))
        elif change_selection == 1:
            return self.change_cnn_layer_size(np.random.randint(low=1, high=16))
        elif change_selection == 2:
            return self.change_add_ann_layer(np.random.randint(low=1, high=16))
        elif change_selection == 3:
            return self.change_add_cnn_layer
        elif change_selection == 4:
            return self.change_epoch(np.random.randint(low=1, high=16))
        elif change_selection == 5:
            return self.change_learning_rate(np.round(np.random.random() * 1e-3, decimals=6))
        elif change_selection == 6:
            return self.change_learning_rate_decay_type
        else:
            return self.change_conv_layer_strides_size(np.random.randint(low=-1, high=1))
