import numpy as np
import airfrans as af
import os, shutil
from lips.dataset.dataSet import DataSet
import copy
from lips.config.configmanager import ConfigManager
from lips.logger.customLogger import CustomLogger

from typing import Union, Callable

class AirfRANSDataSet(DataSet):

    def __init__(self, 
                 config: Union[None, ConfigManager],
                 name: Union[None, str], 
                 attr_names: Union[tuple, None] = None,
                 log_path: Union[str, None] = None,
                 **kwargs
                ):
        super().__init__(name = name)
        self._attr_names = copy.deepcopy(attr_names)
        self.size = 0
        self._inputs = []

        # logger
        self.logger = CustomLogger(__class__.__name__, log_path).logger
        self.config = config

        # number of dimension of x and y (number of columns)
        self._size_x = None
        self._size_y = None
        # self._sizes_x = None  # dimension of each variable
        # self._sizes_y = None  # dimension of each variable
        self._attr_x = kwargs["attr_x"] if "attr_x" in kwargs.keys() else self.config.get_option("attr_x")
        self._attr_y = kwargs["attr_y"] if "attr_y" in kwargs.keys() else self.config.get_option("attr_y")

    def load(self, path: str):
        if not os.path.exists(path):
            raise RuntimeError(f"{path} cannot be found on your computer")
        if not os.path.isdir(path):
            raise RuntimeError(f"{path} is not a valid directory")
        
        index_variable = {
            'x-position': 0,
            'y-position': 1,
            'x-inlet_velocity': 2, 
            'y-inlet_velocity': 3, 
            'distance_function': 4, 
            'x-normals': 5, 
            'y-normmals': 6, 
            'x-velocity': 7, 
            'y-velocity': 8, 
            'pressure': 9, 
            'turbulent_viscosity': 10,
            'surface': 11
        }

        dataset, simulation_names = af.dataset.load(root = path, task = 'scarce', train = True)
        simulation_size = np.array([data.shape[0] for data in dataset])[:, None]
        simulation_names = np.concatenate([np.array(simulation_names)[:, None], simulation_size], axis = 1)
        self.data = {}
        for key in self._attr_names:
            self.data[key] = np.concatenate([sim[:, index_variable[key]] for sim in dataset], axis = 0)
        self.data['simulation_names'] = simulation_names

        self._infer_sizes()

    def _infer_sizes(self):
        """Infer the data sizes"""
        self._size_x = len(self._attr_x)
        self._size_y = len(self._attr_y)

    def get_sizes(self):
        """Get the sizes of the dataset

        Returns
        -------
        tuple
            A tuple of size (size_x, size_y)

        """
        return self._size_x, self._size_y
    
    def extract_data(self) -> tuple:
        """extract the x and y data from the dataset

        Parameters
        ----------
        concat : ``bool``
            If True, the data will be concatenated in a single array.
        Returns
        -------
        tuple
            extracted inputs and outputs
        """
        # init the sizes and everything
        # data = copy.deepcopy(self.data)
        extract_x = np.concatenate([self.data[key][:, None].astype(np.single) for key in self._attr_x], axis = 1)
        extract_y = np.concatenate([self.data[key][:, None].astype(np.single) for key in self._attr_y], axis = 1)
        return extract_x, extract_y

    def save_internal(self, path_out):
        """Save the internal data in a proper format

        Parameters
        ----------
        path_out: output path
            A str to indicate where to save the data.
        """
        full_path_out = os.path.join(os.path.abspath(path_out), self.name)

        if not os.path.exists(os.path.abspath(path_out)):
            os.mkdir(os.path.abspath(path_out))
            self.logger.info(f"Creating the path {path_out} to store the datasets [data will be stored under {full_path_out}]")

        if os.path.exists(full_path_out):
            self.logger.warning(f"Deleting previous run at {full_path_out}")
            shutil.rmtree(full_path_out)

        os.mkdir(full_path_out)
        self.logger.info(f"Creating the path {full_path_out} to store the dataset name {self.name}")

        for attr_nm in self._attr_names:
            np.savez_compressed(f"{os.path.join(full_path_out, attr_nm)}.npz", data = self.data[attr_nm])

if __name__ == '__main__':
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