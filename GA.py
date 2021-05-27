import pprint

from CNN import CNN, create_random_hyper_parameter, generate_random_convolution_parameters
import logging
import numpy as np
import pprint as pp
from logging_config import log_decorator, logger





# @log_decorator
class CNN_GA():
    def __init__(self, population_size):
        self.population_size = population_size
        self.current_generation = 0
        logger.debug('Creating the initial models (generation #0)')
        self.population = []
        gen0_population = []

        for _ in range(population_size):
            modelg0hp = create_random_hyper_parameter()
            modelg0hp['name'] = f'model_gen0_{_}'
            logger.debug(f'New Hyper-parameter created:-\n{pp.pformat(modelg0hp)}')
            logger.debug('Creating and train the model.')
            modelg0 = CNN(modelg0hp, 1)
            gen0_population.append(modelg0)

        print('Done')

    def populate_model_generation(self):
        logger.info('============================================\n' +
                    f'Generation {self.current_generation}' +
                    '============================================\n')

        pass

    def store_generation_models(self):
        self.models[f'gen{self.current_generation}'] = self.population

    @property
    def elite_model(self):
        pass

    @staticmethod
    def strategy_selection(cnn_model) -> str:
        pass

#pprint.pformat(indent=4)
hp = create_random_hyper_parameter()
hp['name'] = 'exampleHP'
pprint.pp(hp)
C = CNN(hp, verbose=1)
pprint.pp(C)
