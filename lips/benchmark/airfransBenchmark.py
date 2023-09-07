"""
Licence:
    Copyright (c) 2021, IRT SystemX (https://www.irt-systemx.fr/en/)
    See AUTHORS.txt
    This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
    If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
    you can obtain one at http://mozilla.org/MPL/2.0/.
    SPDX-License-Identifier: MPL-2.0
    This file is part of LIPS, LIPS is a python platform for power networks benchmarking

"""

import os
import shutil
import warnings
import copy
import json
import time
from typing import Union
import pathlib

import numpy as np

from . import Benchmark
from ..augmented_simulators import AugmentedSimulator
from ..dataset.airfransDataSet import AirfRANSDataSet
from ..evaluation.airfrans_evaluation import AirfRANSEvaluation
from ..utils import NpEncoder



class AirfRANSBenchmark(Benchmark):
    """AirfRANS Benchmark class

    This class allows to benchmark a power grid scenario which are defined in a config file.

    Parameters
    ----------
    benchmark_path : Union[``str``, ``None``], optional
        path to the benchmark, it should be indicated
        if not indicated, the data remains only in the memory
    config_path : Union[``pathlib.Path``, ``str``, ``None``], optional
        path to the configuration file. If config_path is ``None``, the default config file
        present in config module will be used by using the benchmark_name as the section, by default None
    benchmark_name : ``str``, optional
        the benchmark name which is used in turn as the config section, by default "Benchmark1"
    load_data_set : ``bool``, optional
        whether to load the already generated datasets, by default False
    evaluation : Union[``AirfRANSEvaluation``, ``None``], optional
        a ``AirfRANSEvaluation`` instance. If not indicated, the benchmark creates its
        own evaluation instance using appropriate config, by default None
    log_path : Union[``pathlib.Path``, ``str``, ``None``], optional
        path to the logs, by default None

    Warnings
    --------
    An independent class for each benchmark is maybe a better idea.
    This class can be served as the base class for powergrid and a specific class for each benchmark
    can extend this class.
    """
    def __init__(self,
                 benchmark_path: Union[pathlib.Path, str, None],
                 config_path: Union[pathlib.Path, str],
                 benchmark_name: str="Benchmark1",
                 load_data_set: bool=False,
                 evaluation: Union[AirfRANSEvaluation, None]=None,
                 log_path: Union[pathlib.Path, str, None]=None,
                 **kwargs
                 ):
        super().__init__(benchmark_name=benchmark_name,
                         benchmark_path=benchmark_path,
                         config_path=config_path,
                         dataset=None,
                         augmented_simulator=None,
                         evaluation=evaluation,
                         log_path=log_path
                        )

        self.is_loaded=False
        # TODO : it should be reset if the config file is modified on the fly
        if evaluation is None:
            self.evaluation = AirfRANSEvaluation.from_benchmark(self)

        self.env_name = self.config.get_option("env_name")
        self.env = None
        self.training_simulator = None
        self.val_simulator = None
        self.test_simulator = None
        self.test_ood_topo_simulator = None

        self.training_actor = None
        self.val_actor = None
        self.test_actor = None
        self.test_ood_topo_actor = None

        # concatenate all the variables for data generation
        attr_names = self.config.get_option("attr_x") + \
                     self.config.get_option("attr_y")

        self.train_dataset = AirfRANSDataSet(name="train",
                                              attr_names=attr_names,
                                            #   config=self.config,
                                              log_path=log_path
                                              )

        # self.val_dataset = AirfRANSDataSet(name="val",
        #                                     attr_names=attr_names,
        #                                     # config=self.config,
        #                                     log_path=log_path
        #                                     )

        self._test_dataset = AirfRANSDataSet(name="test",
                                              attr_names=attr_names,
                                            #   config=self.config,
                                              log_path=log_path
                                              )

        self._test_ood_topo_dataset = AirfRANSDataSet(name="test_ood_topo",
                                                       attr_names=attr_names,
                                                    #    config=self.config,
                                                       log_path=log_path
                                                       )

        if load_data_set:
            self.load()

    def load(self):
        """
        load the already generated datasets
        """
        if self.is_loaded:
            #print("Previously saved data will be freed and new data will be reloaded")
            self.logger.info("Previously saved data will be freed and new data will be reloaded")
        if not os.path.exists(self.path_datasets):
            raise RuntimeError(f"No data are found in {self.path_datasets}. Have you generated or downloaded "
                               f"some data ?")
        self.train_dataset.load(path=self.path_datasets)
        # self.val_dataset.load(path=self.path_datasets)
        self._test_dataset.load(path=self.path_datasets)
        self._test_ood_topo_dataset.load(path=self.path_datasets)
        self.is_loaded = True

    def evaluate_simulator(self,
                           dataset: str = "all",
                           augmented_simulator: Union[AugmentedSimulator, None] = None,
                           save_path: Union[str, None]=None,
                           save_predictions: bool=False,
                           **kwargs) -> dict:
        """evaluate a trained augmented simulator on one or multiple test datasets

        Parameters
        ----------
        dataset : str, optional
            dataset on which the evaluation should be performed, by default "all"
        augmented_simulator : Union[AugmentedSimulator, None], optional
            An instance of the class augmented simulator, by default None
        save_path : Union[str, None], optional
            the path that the evaluation results should be saved, by default None
        save_predictions: bool
            Whether to save the predictions made by an augmented simulator
            The predictions will be saved at the same directory of the generated data
            # TODO : to save predictions, the directory shoud look like ``benchmark_name\augmented_simulator.name\``
        **kwargs: ``dict``
            additional arguments that will be passed to the augmented simulator
        Todo
        ----
        TODO: add active flow in config file

        Returns
        -------
        dict
            the results dictionary

        Raises
        ------
        RuntimeError
            Unknown dataset selected

        """
        self.augmented_simulator = augmented_simulator
        li_dataset = []
        if dataset == "all":
            li_dataset = [self._test_dataset, self._test_ood_topo_dataset]
            keys = ["test", "test_ood_topo"]
        # elif dataset == "val" or dataset == "val_dataset":
        #     li_dataset = [self.val_dataset]
        #     keys = ["val"]
        elif dataset == "test" or dataset == "test_dataset":
            li_dataset = [self._test_dataset]
            keys = ["test"]
        elif dataset == "test_ood_topo" or dataset == "test_ood_topo_dataset":
            li_dataset = [self._test_ood_topo_dataset]
            keys = ["test_ood_topo"]
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        res = {}
        for dataset_, nm_ in zip(li_dataset, keys):
            # call the evaluate simulator function of Benchmark class
            tmp = self._aux_evaluate_on_single_dataset(dataset=dataset_,
                                                       augmented_simulator=augmented_simulator,
                                                       save_path=save_path,
                                                       save_predictions=save_predictions,
                                                       **kwargs)
            res[nm_] = copy.deepcopy(tmp)

        return res

    def _aux_evaluate_on_single_dataset(self,
                                        dataset: AirfRANSDataSet,
                                        augmented_simulator: Union[AugmentedSimulator, None]=None,
                                        save_path: Union[str, None]=None,
                                        save_predictions: bool=False,
                                        **kwargs) -> dict:
        """Evaluate a single dataset
        This function will evalute a simulator (physical or augmented) using various criteria predefined in evaluator object
        on a ``single test dataset``. It can be overloaded or called to evaluate the performance on multiple datasets

        Parameters
        ------
        dataset : AirfRANSDataSet
            the dataset
        augmented_simulator : Union[AugmentedSimulator, None], optional
            a trained augmented simulator, by default None
        batch_size : int, optional
            batch_size used for inference, by default 32
        active_flow : bool, optional
            whether to compute KCL on active (True) or reactive (False) powers, by default True
        save_path : Union[str, None], optional
            the path where the predictions should be saved, by default None
        save_predictions: bool
            Whether to save the predictions made by an augmented simulator
            The predictions will be saved at the same directory of the generated data
        Returns
        -------
        dict
            the results dictionary
        """
        self.logger.info("Benchmark %s, evaluation using %s on %s dataset", self.benchmark_name,
                                                                            augmented_simulator.name,
                                                                            dataset.name
                                                                            )

        begin_ = time.perf_counter()
        predictions = self.augmented_simulator.predict(dataset, **kwargs)
        end_ = time.perf_counter()
        self.augmented_simulator.predict_time = end_ - begin_

        self.predictions[dataset.name] = predictions
        self.observations[dataset.name] = dataset.data
        self.dataset = dataset

        kwargs["augmented_simulator"] = self.augmented_simulator
        kwargs["dataset"] = dataset
        res = self.evaluation.evaluate(observations=dataset.data,
                                       predictions=predictions,
                                       **kwargs
                                       )

        if save_path:
            if not isinstance(save_path, pathlib.Path):
                save_path = pathlib.Path(save_path)
            save_path = save_path / augmented_simulator.name / dataset.name
            if save_path.exists():
                self.logger.warning("Deleting path %s that might contain previous runs", save_path)
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            with open((save_path / "eval_res.json"), "w", encoding="utf-8") as f:
                json.dump(obj=res, fp=f, indent=4, sort_keys=True, cls=NpEncoder)
            if save_predictions:
                for attr_nm in predictions.keys():
                    np.savez_compressed(f"{os.path.join(save_path, attr_nm)}.npz", data=predictions[attr_nm])
        elif save_predictions:
            warnings.warn(message="You indicate to save the predictions, without providing a path. No predictions will be saved!")

        return res
