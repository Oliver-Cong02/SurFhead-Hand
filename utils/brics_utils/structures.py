import numpy as np
from dataclasses import dataclass
import torch


@dataclass
class MANO: 
    pose: np.ndarray
    shape: np.ndarray
    trans: np.ndarray
    scale: np.ndarray
    fno: np.ndarray

    def __getitem__(self, fno):
        idx = np.where(self.fno == fno)[0][0]
        new_dict = {}
        for key, value in self.__dict__.items():
            if value is not None:
                new_dict[key] = value[idx]
            else:
                new_dict[key] = None
        return MANO(**new_dict)


@dataclass
class Cameras:
    cam_name: np.ndarray
    K: np.ndarray
    distK: np.ndarray
    dist: np.ndarray
    extr: np.ndarray
    fovx: float
    fovy: float
    width: int
    height: int
    world_view_transform: np.ndarray
    projection_matrix: np.ndarray
    full_proj_transform: np.ndarray
    camera_center: np.ndarray

    # def __getitem__(self, idx):
    #     new_dict = {}
    #     for key, value in self.__dict__.items():
    #         new_dict[key] = value[idx]
    #     return Cameras(**new_dict)

    def __getitem__(self, key):
        if isinstance(key, str):
            idx = np.where(self.cam_name == key)[0][0]
        elif isinstance(key, int):
            idx = key
        else:
            raise TypeError("Key must be either an integer or a string representing the camera name.")

        new_dict = {}
        for key, value in self.__dict__.items():
            new_dict[key] = value[idx]
        return Cameras(**new_dict)