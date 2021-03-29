from CNN import CNN
import logging
from logging_config import log_decorator


@log_decorator
class CNN_GA():
    def __init__(self, population_size):
        self.population_size = population_size
