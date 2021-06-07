import os.path
import pprint

from CNN import CNN, create_random_hyper_parameter, generate_random_convolution_parameters, load_data
import logging
import numpy as np
import pandas as pd
import pprint as pp
from typing import Dict, List
from logging_config import log_decorator, logger
from dict2xml import dict2xml


class CNN_GA():
    def __init__(self,
                 population_size: int = 3,
                 model_reruns: int = 2,
                 number_of_models_tobe_changed_based_on_training_time: int = 1):
        _, __, self.input_shape, self.output_size = load_data(1)
        self.number_of_models_tobe_changed_based_on_training_time = number_of_models_tobe_changed_based_on_training_time
        self.number_of_models_per_generation = population_size
        self.model_reruns = model_reruns
        self.current_generation = 0
        self.metrics = pd.DataFrame(
            columns=['test_Accuracy', 'train_Accuracy', 'training_time', 'prev_model', 'generation'])
        logger.debug('Creating the initial models (generation #0)')
        self.models = dict()
        self.current_generation_models = []

        for _ in range(self.number_of_models_per_generation):
            modelg0hp = create_random_hyper_parameter(output_size=self.output_size,
                                                      number_of_cnn_layers=4,
                                                      number_of_ann_layers=4)
            modelg0hp = CNN.change_name_to(modelg0hp, f'model_gen0_{_}')
            modelg0hp['prev_model'] = 'new'
            self.current_generation_models.append(modelg0hp)
            logger.debug(f'New Hyper-parameter created:-\n{pp.pformat(modelg0hp)}')
        self.train_current_generation()
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
            try:
                model_runs = [CNN(model, verbose=1) for _ in range(self.model_reruns)]
            except Exception as error:
                logger.error(error)
                # revert Changes
                prev_model = model['prev_model']
                model = self.models[prev_model]
                model = CNN.add_change_log(model, f'Reverted to model {prev_model} due to an exception on training.')
                model_name = model['name']
                model_runs = [CNN(model, verbose=1) for _ in range(self.model_reruns)]

            logger.info(f'Training model {model_name} completed')
            self.metrics.loc[model_name, 'test_Accuracy'] = np.min([cnn.accuracy[0] for cnn in model_runs])
            self.metrics.loc[model_name, 'train_Accuracy'] = np.min([cnn.accuracy[1] for cnn in model_runs])
            self.metrics.loc[model_name, 'training_time'] = np.max([cnn.Training_time for cnn in model_runs])
            self.metrics.loc[model_name, 'over-fit'] = np.any([cnn.is_over_fitted for cnn in model_runs])
            self.metrics.loc[model_name, 'prev_model'] = model['prev_model']
            self.metrics.loc[model_name, 'generation'] = self.current_generation
            model['layers_input_output_shape'] = [ f'layer.name: {layer.input_shape} --- {layer.output_shape}'
                                                  for layer in model_runs[0].model.layers]
            self.save_model(model)
            logger.info(f'Performance results for {model_name}:-\n{self.metrics.loc[model_name, :]}')
        logger.info(f'Generation {self.current_generation} Training completed.\n------------------\n')

    def save_model(self, model_hp: Dict):
        model_name = model_hp['name']
        self.models[model_name] = model_hp
        file_path = os.path.join('models', f'{model_name}.xml')
        with open(file_path, 'w') as file:
            file.writelines(dict2xml(
                model_hp,
                wrap='model_hyper_parameters',
                indent="\t"))
            logger.debug(f'{model_name} was saved in {file_path}')

    def next_generation_models(self):
        self.current_generation += 1
        logger.info('============================================\n' +
                    f'Generation {self.current_generation}' +
                    '============================================\n')
        # Elite Selection
        elite = self.elite_model(self.current_generation - 1)
        next_gen_models = [elite]

        # slowest Training Time changes
        n = self.number_of_models_tobe_changed_based_on_training_time
        slow_models = self.top_n_slowest_models(self.current_generation - 1, n)
        for _, slow_model in enumerate(slow_models):
            new_model = CNN.change_for_slow_training_time(slow_model)
            new_model = CNN.change_name_to(self.models[new_model], f'model_gen{self.current_generation}_{_}')
            next_gen_models.append(new_model)

        # Fix Under-fitting and over-fitting for the rest
        prev_gen_model_names = set([model['name'] for model in self.current_generation_models])
        elite_set = set(elite['name'])
        slow_models_names = set([model['name'] for model in slow_models])
        under_fitted_models = list(prev_gen_model_names - elite_set - slow_models_names)
        for _, prev_gen_model in enumerate(under_fitted_models):
            model_hp = self.models[prev_gen_model]
            if self.metrics.loc[prev_gen_model, 'over-fit'] ==1:
                new_model = CNN.change_for_over_fit(model_hp, self.input_shape)
            else:
                new_model = CNN.change_for_under_fitting(model_hp, self.input_shape, self.output_size)
            new_model = CNN.change_name_to(new_model, f'model_gen{self.current_generation}_{_}')
            next_gen_models.append(new_model)

        # Run the New Generation models
        self.current_generation_models = next_gen_models
        self.train_current_generation()

    def elite_model(self, generation: int) -> Dict:
        generation_metrics = self.metrics[self.metrics.loc[:, 'generation'] == generation]
        generation_metrics.sort_values(by=['test_Accuracy'], ascending=False, inplace=True)
        elite_model_name = generation_metrics.index.values[0]
        logger.info(f'Model {elite_model_name} is selected as the Elite model.')
        return self.models[elite_model_name]

    def top_n_slowest_models(self, generation: int, n: int = 3) -> List[Dict]:
        generation_metrics = self.metrics[self.metrics.loc[:, 'generation'] == generation]
        generation_metrics.sort_values(by=['training_time'], ascending=True, inplace=True)
        elite_model_name = generation_metrics.index.values[0:n - 1]
        return [self.models[name] for name in elite_model_name]


ga = CNN_GA(population_size=3,
            model_reruns=2,
            number_of_models_tobe_changed_based_on_training_time=1)

for _ in range(10):
    ga.train_current_generation()
    ga.metrics.to_csv(os.path.join(os.getcwd(), 'model_metrics.csv'))
    ga.next_generation_models()


print(ga.metrics)
