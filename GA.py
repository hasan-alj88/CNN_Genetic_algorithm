import pprint

from CNN import CNN, create_random_hyper_parameter, generate_random_convolution_parameters
import logging
import numpy as np
import pandas as pd
import pprint as pp
from logging_config import log_decorator, logger


class CNN_GA():
    def __init__(self, population_size:int=10, model_reruns:int= 3):
        self.number_of_models_per_generation = population_size
        self.model_reruns = model_reruns
        self.current_generation = 0
        self.metrics = pd.DataFrame(
            columns=['test_Accuracy', 'train_Accuracy', 'training_time', 'prev_model', 'generation'])
        logger.debug('Creating the initial models (generation #0)')
        self.models = dict()
        self.current_generation_models = []

        for _ in range(population_size):
            modelg0hp = create_random_hyper_parameter()
            modelg0hp['name'] = f'model_gen0_{_}'
            modelg0hp['prev_model'] = 'new'
            self.current_generation_models.append(modelg0hp)
            logger.debug(f'New Hyper-parameter created:-\n{pp.pformat(modelg0hp)}')
        logger.debug('Creating and train the model.')

        print('Done')

    def __len__(self):
        return len(self.models.keys())

    def __repr__(self):
        s = '< Model hyper-parameter Genetic algorithm simulator object,\n'
        s += f'Models simulated :{len(self)}, check models stats @ .metrics >'
        return s

    def train_current_generation(self):
        logger.info(f'Training generation {self.current_generation}')
        for model in self.current_generation_models:
            model_name = model['name']
            logger.info(f'Training model {model_name}.')
            model_runs = [CNN(model) for _ in range(self.model_reruns)]
            logger.info(f'Training model {model_name} completed')
            self.metrics.loc[model_name, 'test_Accuracy'] = np.min([cnn.accuracy[0] for cnn in model_runs])
            self.metrics.loc[model_name, 'train_Accuracy'] = np.min([cnn.accuracy[1] for cnn in model_runs])
            self.metrics.loc[model_name, 'training_time'] = np.max([cnn.Training_time for cnn in model_runs])
            self.metrics.loc[model_name, 'prev_model'] = model['prev_model']
            self.metrics.loc[model_name, 'generation'] = self.current_generation
            logger.info(f'Performance results for {model_name}:-\n{self.metrics.loc[model_name,:]}')
        logger.info(f'Generation {self.current_generation} Training completed.\n------------------\n')

    def next_generation_models(self):
        logger.info('============================================\n' +
                    f'Generation {self.current_generation}' +
                    '============================================\n')

        pass

    def store_generation_models(self):
        for model in self.current_generation_models:
            self.models[model['name']] = model

    @property
    def elite_model(self):
        pass


# pprint.pformat(indent=4)
ga = CNN_GA(10)
