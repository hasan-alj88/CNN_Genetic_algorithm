import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import List


def scale(in_data):
    return (in_data - np.min(in_data)) / np.max(in_data)


def load_data():
    data = pd.read_csv('pima-indians-diabetes-data.csv', header=None)
    data = data.sample(frac=1)
    for col in data.columns[:-1]:
        data.loc[:, col] = scale(data.loc[:, col])
    split = int(len(data) * 0.9)
    train_x = data.iloc[:split, 0:7].values
    train_y = data.iloc[:split, 8].values.reshape((-1, 1))
    test_x = data.iloc[split:, 0:7].values
    test_y = data.iloc[split:, 8].values.reshape((-1, 1))
    return (train_x, train_y), (test_x, test_y)


class AnyDice:
    def __init__(self, probabilities: List[float]):
        assert np.isclose(np.sum(probabilities), 1)
        self.boundaries = np.append([0], np.add.accumulate(probabilities))

    @property
    def roll(self):
        roll = np.random.random()
        for p in range(len(self.boundaries) - 1):
            if self.boundaries[p] < roll < self.boundaries[p + 1]:
                return p


class NN:
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    input_shape = train_x[0].shape
    output_shape = max(train_y[0].shape)
    classification = len(np.unique(train_y))

    def __init__(self, hidden_layers_nodes, dropout_layers=None,
                 learning_rate=0.001, epochs=32, batch_size=12, verbose=2, regression=False):

        dropout_layers = np.ones(len(hidden_layers_nodes)) if dropout_layers is None else dropout_layers
        self._hyper_parameters = {'hidden layers': hidden_layers_nodes,
                                  'dropout layers': dropout_layers,
                                  'learning rate': learning_rate,
                                  'epochs': epochs,
                                  'batch size': batch_size
                                  }
        # Creating the Neural network model
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(hidden_layers_nodes[0], input_shape=self.input_shape, activation='relu'))
        if 1 > dropout_layers[0] > 0:
            self.model.add(layers.Dropout(dropout_layers[0]))
        for hidden_layer, dropout_layer in zip(hidden_layers_nodes[1:], dropout_layers[1:]):
            self.model.add(layers.Dense(hidden_layer, activation='relu'))
            if 1 > dropout_layer > 0:
                self.model.add(layers.Dropout(dropout_layer))
        self.classification_type = 1 if regression else (3 if self.classification > 2 else 2)
        # Output layer
        output_layer_choices = {
            # Regression
            1: layers.Dense(1, activation=None),
            # Binary classification
            2: layers.Dense(1, activation=tf.keras.activations.sigmoid),
            # multi-classification
            3: layers.Dense(self.classification, activation=tf.keras.activations.softmax)
        }
        self.model.add(output_layer_choices[self.classification_type])
        print(f'\nNeural network created with the following layers:-{self.print_layers}')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        print(f'Training the Neural network')
        self.History = self.model.fit(x=self.train_x, y=self.train_y,
                                      epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.train_x, self.train_y),
                                      shuffle=True, verbose=verbose)

    @property
    def print_layers(self):
        layers_string = f'\n{self.input_shape} >> '
        for hid, drop in zip(self.hyper_parameters['hidden layers'],
                             self.hyper_parameters['dropout layers']):
            layers_string += f'{hid} :{drop:0.00f} | '
        layers_string += f" >> 2\nepochs = {self.hyper_parameters['epochs']}\n"
        layers_string += f"batch size = {self.hyper_parameters['batch size']}\n"
        layers_string += f"learning rate = {self.hyper_parameters['learning rate']}"
        return layers_string

    def __repr__(self):
        summary_string = '\n---------------------'
        summary_string += self.print_layers
        loss1, accuracy1 = self.evaluate_train
        loss2, accuracy2 = self.evaluate_test
        summary_string += f'\nTraining Data:-\nloss = {loss1}\naccuracy = {accuracy1 * 100:00.00f}%'
        summary_string += f'\nTest Data:-\nloss = {loss2}\naccuracy = {accuracy2 * 100:00.000f}%'
        summary_string += '\n---------------------\n\n'
        return summary_string

    @property
    def hyper_parameters(self):
        return self._hyper_parameters

    @property
    def evaluate_train(self):
        return self.model.evaluate(x=self.train_x, y=self.train_y, verbose=0)

    @property
    def evaluate_test(self):
        return self.model.evaluate(x=self.test_x, y=self.test_y, verbose=0)

    def prediction(self, data):
        assert data[0].shape == self.input_shape
        if self.classification_type = 1:
            return [self.model.predict(_) for _ in data.reshape(-1,self.classification)]
        return [1 if (self.model.predict(_.reshape(-1, 7))) > 0.5 else 0 for _ in data]

    @property
    def prediction_train_data(self):
        return self.prediction(self.train_x)

    @property
    def prediction_test_data(self):
        return self.prediction(self.test_x)

    @property
    def get_accuracy(self):
        _, test_acc = self.evaluate_train
        _, train_acc = self.evaluate_train
        return test_acc, train_acc

    def plot_accuracy(self):
        plt.plot(self.History.history['accuracy'])
        plt.plot(self.History.history['val_accuracy'])
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        plt.plot(self.History.history['loss'])
        plt.plot(self.History.history['val_loss'])
        plt.grid(True)
        plt.show()

    def plot_nn_confusion_matrix(self):
        cm_train = confusion_matrix(self.train_y, self.prediction_train_data)
        # cm_train /= np.max(cm_train)
        cm_test = confusion_matrix(self.test_y, self.prediction_test_data)
        # cm_test /= np.max(cm_test)
        plt.subplot(1, 2, 1)
        plt.imshow(cm_train)
        plt.title('\nTraining dataset\nconfusion matrix')
        plt.ylabel('prediction')
        plt.xlabel('Ground')
        plt.subplot(1, 2, 2)
        plt.imshow(cm_test)
        plt.title('\nTesting dataset\nconfusion matrix')
        plt.ylabel('prediction')
        plt.xlabel('Ground')
        print(f'training data accuracy = {100 * np.sum(cm_train.diagonal()) / np.sum(cm_train.reshape(-1, 1)):2.2f}%')
        print(f'Testing data accuracy = {100 * np.sum(cm_test.diagonal()) / np.sum(cm_test.reshape(-1, 1)):2.2f}%')
        plt.tight_layout()
        plt.show()

    def plot_weights(self):
        weights = np.array(self.model.get_weights(), dtype=object).reshape(-1, 2)
        for layer, weight in enumerate(weights):
            print(f'Layer {layer}')
            print(f'weight shape = {weight[0].shape}\nbias shape = {weight[1].shape}')
            plt.subplot(2, 1, 1)
            plt.imshow(weight[0])
            plt.subplot(2, 1, 2)
            plt.imshow(weight[1].reshape(1, -1))
            plt.tight_layout()
            plt.show()

    def print_layers_output(self, in_data):
        print(f'Data in:\n{in_data}')
        weights = np.array(self.model.get_weights(), dtype=object).reshape(-1, 2)
        for layer, weight in enumerate(weights):
            print(f'Layer {layer}')
            print(np.dot(weight[0], in_data) + weight[1])

    # Mutations
    @classmethod
    def with_hyper_parameters(cls, hyper_parameters, verbose=0):
        return cls(hidden_layers_nodes=hyper_parameters['hidden layers'],
                   dropout_layers=hyper_parameters['dropout layers'],
                   learning_rate=hyper_parameters['learning rate'],
                   epochs=hyper_parameters['epochs'],
                   batch_size=hyper_parameters['batch size'],
                   verbose=verbose)

    @property
    def mutation_add_layer(self):
        hyp = self.hyper_parameters
        n = np.random.randint(2, 128)
        hid_layer = np.array(hyp['hidden layers']).astype(int)
        drop_layer = np.array(hyp['dropout layers']).astype(float)
        hid_layer = np.append(hid_layer, n)
        drop_layer = np.append(drop_layer, 0.0)
        hyp['hidden layers'] = hid_layer
        hyp['dropout layers'] = drop_layer
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    @property
    def mutation_remove_layer(self):
        if len(self.hyper_parameters['hidden layers']) < 3:
            return self
        hyp = self.hyper_parameters
        n = np.random.randint(len(hyp['hidden layers']))
        hid_layer = np.array(list(hyp['hidden layers'])).astype(int)
        drop_layer = np.array(list(hyp['dropout layers'])).astype(float)
        del hid_layer[n]
        del drop_layer[n]
        hyp['hidden layers'] = hid_layer
        hyp['dropout layers'] = drop_layer
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    def mutation_dropout(self, change):
        hyp = self.hyper_parameters
        n = np.random.randint(len(hyp['dropout layers']))
        drop_layer = np.array(hyp['dropout layers']).astype(int)
        drop_layer[n] += drop_layer[n] + change if 0.0 < (drop_layer[n] + change) < 1.0 else drop_layer[n]
        hyp['dropout layers'] = drop_layer
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    def mutation_nodes(self, change):
        hyp = self.hyper_parameters
        n = np.random.randint(len(hyp['hidden layers']))
        hid_layer = np.array(list(hyp['hidden layers'])).astype(int)
        hid_layer[n] += change if hid_layer[n] + change > 1 else 0
        hyp['hidden layers'] = hid_layer
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    def mutation_epochs(self, change):
        hyp = self.hyper_parameters
        hyp['epochs'] += change
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    def mutation_batch_size(self, change):
        hyp = self.hyper_parameters
        hyp['batch size'] += int(change if hyp['batch size'] + change > 0 else 0)
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    def mutation_learning_rate(self, change):
        hyp = self.hyper_parameters
        hyp['learning rate'] += change if 0 < hyp['learning rate'] + change < 0.1 else 0
        return NN.with_hyper_parameters(hyper_parameters=hyp, verbose=0)

    @classmethod
    def random_nn(cls, verbose=0):
        hidden_layers_random = np.random.randint(1, 4)
        return cls(hidden_layers_nodes=np.array([np.random.randint(4, 128)
                                                 for _ in range(hidden_layers_random)]).astype(int),
                   dropout_layers=np.zeros(hidden_layers_random),
                   learning_rate=0.001,
                   epochs=np.random.randint(32, 64),
                   batch_size=np.random.randint(1, 32),
                   verbose=verbose)

    @property
    def over_fitting_mutation(self):
        print('The Neural network Over Fitted')
        roll = AnyDice([0.4, 0.1, 0.1, 0.3, 0.1]).roll
        if roll == 0:
            print('Thus increasing Dropout layers effects')
            return self.mutation_dropout(-np.random.random() * 0.2)
        elif roll == 1:
            print('Thus decreasing epochs')
            return self.mutation_epochs(-np.random.randint(8, 32))
        elif roll == 2:
            print('Thus decreasing batch size')
            return self.mutation_batch_size(-np.random.randint(8, 32))
        elif roll == 3:
            print('Thus decreasing random hidden layer size')
            return self.mutation_nodes(-np.random.randint(1, 32))
        else:
            print('Thus removing a hidden layer')
            return self.mutation_remove_layer

    @property
    def under_fitting_mutation(self):
        print('The Neural network UnderFitted')
        roll = AnyDice([0.05, 0.3, 0.1, 0.3, 0.15, 0.1]).roll
        if roll == 0:
            print('Thus decreasing Dropout layers effects')
            return self.mutation_dropout(np.random.random() * 0.2)
        elif roll == 1:
            print('Thus increasing epochs')
            return self.mutation_epochs(np.random.randint(8, 32))
        elif roll == 2:
            print('Thus increasing batch size')
            return self.mutation_batch_size(np.random.randint(1, 32))
        elif roll == 3:
            print('Thus increasing random hidden layer size')
            return self.mutation_nodes(np.random.randint(1, 32))
        elif roll == 4:
            print('Thus adjusting learning rate')
            return self.mutation_learning_rate(np.random.randint(-1000, 1000) * 0.0001 * 0.001)
        else:
            print('Thus adding a hidden layer')
            return self.mutation_add_layer

    @property
    def fitting_mutation(self):
        print('Fitting The Neural network')
        roll = AnyDice([0.5, 0.1, 0.3, 0.1]).roll
        if roll == 0:
            print('By increasing epochs')
            return self.mutation_epochs(4)
        elif roll == 1:
            print('By increasing batch size')
            return self.mutation_batch_size(1)
        elif roll == 2:
            print('By increasing random hidden layer size')
            return self.mutation_nodes(1)
        else:
            print('By decreasing batch size')
            return self.mutation_batch_size(-1)

    @property
    def fitting_status(self) -> str:
        test_acc, train_acc = self.get_accuracy
        if train_acc > 0.95 and train_acc > test_acc:
            return 'Over fit'
        elif test_acc < 0.9 or train_acc < 0.9:
            return 'Under fit'  # under fit
        else:
            return 'Normal'  # normal


class NNGA:
    def __init__(self, population=5):
        print('creating random Neural networks:-')
        self.generation = 0
        self.population = [NN.random_nn(verbose=0) for _ in range(population)]
        self.print_generation()

    def next_generation(self):
        new_generation = [self.population[self.best_nn_index]]  # elite selection
        pop = self.population
        del pop[self.best_nn_index]  # no mutating for the elite
        print('mutating Neural networks...')
        for nn in pop:
            print(nn)
            fitting = nn.fitting_status
            if fitting == 'Over fit':
                new_generation.append(nn.over_fitting_mutation)
            elif fitting == 'Under fit':
                new_generation.append(nn.under_fitting_mutation)
            else:
                new_generation.append(nn.fitting_mutation)
        self.population = new_generation
        self.generation += 1
        self.print_generation()

    @property
    def best_nn_index(self):
        return np.argmax([nn.get_accuracy[0] for nn in self.population])

    @property
    def best_nn(self):
        return self.population[self.best_nn_index]

    def print_generation(self):
        print(f'\nGeneration #{self.generation}')
        for nn in self.population:
            print(nn)
        print(f'Best of this generation...\n{self.best_nn}')
