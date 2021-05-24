from CNN import CNN, create_random_hyper_parameter
import logging
import numpy as np

from logging_config import log_decorator, logger

@log_decorator
class AnyDice:
    def __init__(self, probabilities):
        if not np.isclose(np.sum([0, probabilities]), 1.0):
            raise ValueError('sum of probabilities must equal 1.0')
        probabilities.insert(0, 0)
        self.boundaries = probabilities.ufunc.accumulate()

    @log_decorator
    @property
    def roll(self) -> int:
        the_roll = np.random.random()
        for interval, (lower, upper) in enumerate(zip(self.boundaries[:-1], self.boundaries[1:])):
            if lower <= the_roll <= upper:
                return interval
        else:
            return len(self.boundaries) - 1





@log_decorator
class CNN_GA():
    def __init__(self, population_size):
        self.population_size = population_size
        self.current_generation = 0
        logger.debug('Creating the initial models (generation #0)')
        self.population = []
        gen0_population = []

        for _ in range(population_size):
            modelg0 = create_random_hyper_parameter()


        self.generate_models()

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
