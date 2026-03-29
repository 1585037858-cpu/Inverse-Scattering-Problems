import os
import sys
import numpy as np
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.doi_utils import DOIUtils
from utils.plot_utils import PlotUtils


class Scatterer:

    def __init__(self, problem, inverse_type, scatterer_params: list):
        self.problem = problem
        self.inverse_type = inverse_type
        self.scatterer_params = scatterer_params
        self.grid_positions = DOIUtils.get_grid_centroids(problem)
        m = len(self.grid_positions[0])

        if self.problem == "inverse" and self.inverse_type == "ratio":
            self.scatterer = np.zeros((m, m), dtype=complex)
        else:
            self.scatterer = np.ones((m, m), dtype=complex)

    def get_permittivity(self, param):
        assert self.problem == "forward" or self.problem == "inverse"
        if self.problem == "inverse" and self.inverse_type == "ratio":
            epsilon_R = np.real(param["permittivity"])
            epsilon_I = np.imag(param["permittivity"])
            value = epsilon_I / np.sqrt(epsilon_R)
        else:
            value = param["permittivity"]
        return value

    def circle_scatterer(self, param):
        value = self.get_permittivity(param)
        self.scatterer[(self.grid_positions[0] - param["center_x"]) ** 2 + (self.grid_positions[1] - param["center_y"]) ** 2 <= param["size"] ** 2] = value

    def square_scatterer(self, param):
        value = self.get_permittivity(param)
        mask = ((self.grid_positions[0] <= param["center_x"] + param["size"]) & (self.grid_positions[0] >= param["center_x"] - param["size"]) &
                (self.grid_positions[1] <= param["center_y"] + param["size"]) & (self.grid_positions[1] >= param["center_y"] - param["size"]))
        self.scatterer[mask] = value

    def rectangle_scatterer(self, param):
        value = self.get_permittivity(param)
        mask = ((self.grid_positions[0] <= param["center_x"] + param["size1"]) & (self.grid_positions[0] >= param["center_x"] - param["size1"]) &
                (self.grid_positions[1] <= param["center_y"] + param["size2"]) & (self.grid_positions[1] >= param["center_y"] - param["size2"]))
        self.scatterer[mask] = value

    def polygon_scatterer(self, param):
        value = self.get_permittivity(param)
        poly_verts = param["vertices"]
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = self.grid_positions
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        path = Path(poly_verts)
        grid = path.contains_points(points)
        m = len(self.grid_positions[0])
        grid = grid.reshape((m, m))
        print(grid)
        self.scatterer[grid] = value

    def generate(self):
        for param in self.scatterer_params:
            if param["shape"] == "circle":
                self.circle_scatterer(param)
            elif param["shape"] == "square":
                self.square_scatterer(param)
            elif param["shape"] == "rectangle":
                self.rectangle_scatterer(param)
            elif param["shape"] == "polygon":
                self.polygon_scatterer(param)
            else:
                raise ValueError("Invalid scatterer shape")
        return self.scatterer     # 散射体在建设时，进行for循环，获取到第一个散射体的介电常数值后，赋给self.scatterer，后续再进行第二个散射体赋值


if __name__ == '__main__':

    scatterer_params = [
        {
            "shape": "circle",
            "center_x": 0.3,
            "center_y": 0.3,
            "size": 0.15,
            "permittivity": 3.4 + 0.25j
        },
        {
            "shape": "square",
            "center_x": -0.3,
            "center_y": 0.3,
            "size": 0.15,
            "permittivity": 77 + 7j
        }
    ]

    sc = Scatterer("forward", None, scatterer_params)
    forward_scatterer = sc.generate()       # nd.array (400×400) 包含两个 散射体 介电常数数据

    sc = Scatterer("inverse", None, scatterer_params)
    inverse_scatterer = sc.generate()       # nd.array (50×50) 包含两个 散射体 介电常数数据

    PlotUtils.view_scatterer(forward_scatterer, inverse_scatterer)        # 查看前向跟反向的散射体介电常常数

