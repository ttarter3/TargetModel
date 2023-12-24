from typing import Callable
import numpy as np
import numpy.typing as npt
from numpy import ScalarType


class Model3D:

    def __init__(self, object_name: str
                 , vertices: np.ndarray[npt.Any, np.dtype[+ScalarType]]
                 , edges: np.ndarray[npt.Any, np.dtype[+ScalarType]]
                 , general_description_of_reference_frame: str) -> object:
        """
        Generic Model for hosting a 3d Object
        :param object_name: UniqueId for the object of interest. Multiple objects of similar builds would have 'airplane1', 'airplane2'
        :param vertices: xyz coordinate system of points
        :param edges: 0 based, reference of vertices to connect to create triangles
        :param general_description_of_reference_frame: a useful description so that someone know the units and reference of the nose of the object
        """
        self.object_name_ = object_name
        self.vertices_ = vertices
        self.edges_ = edges
        self.general_description_of_reference_frame_ = general_description_of_reference_frame
