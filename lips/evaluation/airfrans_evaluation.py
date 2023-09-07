"""
Usage:
    Grid2opSimulator implementing powergrid physical simulator
Licence:
    copyright (c) 2021-2022, IRT SystemX and RTE (https://www.irt-systemx.fr/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking
"""
from typing import Union
from collections.abc import Iterable
import numpy as np

from lips.config.configmanager import ConfigManager
from lips.evaluation import Evaluation
from lips.logger import CustomLogger
from lips.evaluation.utils import metric_factory

import airfrans as af
from scipy.stats import spearmanr

class AirfRANSEvaluation(Evaluation):
    """Evaluation of the AirfRANS specific metrics

    It is a subclass of the Evaluation class

    Parameters
    ----------
    config, optional
        an object of `ConfigManager` class
    config_path, optional
        _description_, by default None
    # scenario, optional
    #     one of the Power Grid Scenario names, by default None
    log_path, optional
        path where the log should be maintained, by default None
    """
    def __init__(self,
                 data_path: str,
                 config: Union[ConfigManager, None]=None,
                 config_path: Union[str, None]=None,
                 scenario: Union[str, None]=None,
                 log_path: Union[str, None]=None
                 ):
        super(AirfRANSEvaluation,self).__init__(config=config,
                                                 config_path=config_path,
                                                 config_section=scenario,
                                                 log_path=log_path
                                                 )

        self.data_path = data_path
        # self.eval_dict = self.config.get_option("eval_dict")
        # self.eval_params = self.config.get_option("eval_params")
        # self.eval_crit_args = self.config.get_option("eval_crit_args")

        self.logger = CustomLogger(__class__.__name__, self.log_path).logger

        # scenario_params=self.config.get_option("env_params")
        # self.simulator = GetfemSimulator(**scenario_params)
        # self.simulator.build_model()

    def from_batch_to_simulation(self, data):
        sim_data = {}
        keys = list(set(data.keys()) - set(['simulation_names']))
        for key in keys:
            sim_data[key] = []
            ind = 0
            for n in range(data['simulation_names'].shape[0]):           
                sim_data[key].append(data[key][ind:(ind + int(data['simulation_names'][n, 1]))])
                ind += int(data['simulation_names'][n, 1])
        sim_data['simulation_names'] = data['simulation_names'][:, 0]

        return sim_data

    @classmethod
    def from_benchmark(cls,
                       benchmark: "AirfRANSBenchmark",
                      ):
        """ Intialize the evaluation class from a benchmark object

        Parameters
        ----------
        benchmark
            a benchmark object

        Returns
        -------
        PneumaticEvaluation
        """
        return cls(config=benchmark.config, log_path=benchmark.log_path)

    def evaluate(self,
                 observations: dict,
                 predictions: dict,
                 save_path: Union[str, None]=None) -> dict:
        """The main function which evaluates all the required criteria noted in config file

        Parameters
        ----------
        dataset
            DataSet object including true observations used to evaluate the predictions
        predictions
            predictions obtained from augmented simulators
        save_path, optional
            path where the results should be saved, by default None
        """
        # call the base class for generic evaluations
        super().evaluate(observations, predictions, save_path)

        for cat in self.eval_dict.keys():
            self._dispatch_evaluation(cat)

        # TODO: save the self.metrics variable
        if save_path:
            pass

        return self.metrics

    def _dispatch_evaluation(self, category: str):
        """
        This helper function select the evaluation function with respect to the category

        In AirfRANS case, the OOD generalization evaluation is performed using `Benchmark` class
        by iterating over all the datasets

        Parameters
        ----------
        category: `str`
            the evaluation criteria category, the values could be one of the [`ML`, `Physics`]
        """
        if category == self.MACHINE_LEARNING:
            if self.eval_dict[category]:
                self.evaluate_ml()
        if category == self.PHYSICS_COMPLIANCES:
            if self.eval_dict[category]:
                self.evaluate_physics()
        if category == self.INDUSTRIAL_READINESS:
            raise Exception("Not done yet, sorry")

    def evaluate_ml(self):
        """
        Verify Pneumatic Machine Learning metrics
        """
        metric_val_by_name = self.metrics[self.MACHINE_LEARNING]
        for metric_name in self.eval_dict[self.MACHINE_LEARNING]:
            metric_fun = metric_factory.get_metric(metric_name)
            metric_val_by_name[metric_name] = {}
            for nm_, pred_ in self.predictions.items():
                true_ = self.observations[nm_]
                tmp = metric_fun(true_, pred_)
                if isinstance(tmp, Iterable):
                    metric_val_by_name[metric_name][nm_] = [float(el) for el in tmp]
                    self.logger.info("%s for %s: %s", metric_name, nm_, tmp)
                else:
                    metric_val_by_name[metric_name][nm_] = float(tmp)
                    self.logger.info("%s for %s: %.2E", metric_name, nm_, tmp)

    def evaluate_physics(self):
        """
        function that evaluates physical criteria on given observations and may rely on the physical solver
        """

        pred_data = self.from_batch_to_simulation(self.predictions)
        true_coefs = []
        coefs = []
        rel_err = []

        for n, simulation_name in enumerate(pred_data['simulation_names']):
            simulation = af.Simulation(root = self.data_path, name = simulation_name)

            simulation.velocity = np.concatenate([pred_data['x-velocity'][n][:, None], pred_data['y-velocity'][n][:, None]], axis = 1)
            simulation.pressure = pred_data['pressure'][n]
            simulation.nu_t = pred_data['turbulent_viscosity'][n]
            coefs.append(simulation.force_coefficient())
            rel_err.append(simulation.coefficient_relative_error())
            true_coefs.append(simulation.force_coefficient(reference = True))
        rel_err = np.array(rel_err)
        
        spear_drag = np.array([coefs[n][0][0] for n in range(len(coefs))])
        spear_true_drag = np.array([true_coefs[n][0][0] for n in range(len(true_coefs))])
        spear_lift = np.array([coefs[n][1][0] for n in range(len(coefs))])
        spear_true_lift = np.array([true_coefs[n][1][0] for n in range(len(true_coefs))])

        spear_coefs = (spearmanr(spear_drag, spear_true_drag)[0], spearmanr(spear_lift, spear_true_lift)[0])

        mean_rel_err, std_rel_err = rel_err.mean(axis = 0), rel_err.std(axis = 0)
        self.logger.info('The mean relative absolute error for the drag coefficient is: {:.3f}'.format(mean_rel_err[0]))
        self.logger.info('The standard deviation of the relative absolute error for the drag coefficient is: {:.3f}'.format(std_rel_err[0]))
        self.logger.info('The mean relative absolute error for the lift coefficient is: {:.3f}'.format(mean_rel_err[1]))
        self.logger.info('The standard deviation of the relative absolute error for the lift coefficient is: {:.3f}'.format(std_rel_err[1]))
        self.logger.info('The spearman correlation for the drag coefficient is: {:.3f}'.format(spear_coefs[0]))
        self.logger.info('The spearman correlation for the lift coefficient is: {:.3f}'.format(spear_coefs[1]))

        return {'target_coefficients': true_coefs, 'predicted_coefficients': coefs, 'relative absolute error': rel_err}

if __name__ == '__main__':
    from lips.dataset.airfransDataSet import AirfRANSDataSet
    evaluation = AirfRANSEvaluation(data_path = '/home/florent/Dataset', log_path = '/home/florent/LIPS/log_eval')
    attr_names = (
        'x-position',
        'y-position',
        'x-inlet_velocity', 
        'y-inlet_velocity', 
        'distance_function', 
        'x-normals', 
        'y-normmals', 
        'x-velocity', 
        'y-velocity', 
        'pressure', 
        'turbulent_viscosity',
        'surface'
    )
    attr_x = attr_names[:7]
    attr_y = attr_names[7:]
    my_dataset = AirfRANSDataSet(config = None, name = 'train', attr_names = attr_names, log_path = '/home/florent/LIPS/log', attr_x = attr_x, attr_y = attr_y)
    my_dataset.load(path = '/home/florent/Dataset')
    evaluation.evaluate(observations = None, predictions = my_dataset.data)