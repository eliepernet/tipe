import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.setrecursionlimit(10000)

class AneurysmData:
    def __init__(self, file, w, h):
        mesh = pv.read(file)
        slice_plane = mesh.slice(normal='z')

        edges = slice_plane.extract_feature_edges(boundary_edges=True)
        contour_points = np.array(edges.points)
        contour_points = np.array([[cell[1], cell[0] + 10] for cell in contour_points])

        self.file = file
        self.w = w
        self.h = h
        self.s_h = 14 / self.h
        self.s_w = 20 / self.w

        self.grid = np.array([[0 for _ in range(self.w)] for _ in range(self.h)])

        for point in contour_points:
            nearest_cell_x = int(np.floor(point[1] / self.s_w))
            nearest_cell_y = int(np.floor(point[0] / self.s_h))
            self.grid[nearest_cell_y][nearest_cell_x] = 1

        self.__filling(int(h / 2), int(w / 2))
        self.grid[int(h / 2)][int(w / 2)] = 2
        self.__filling_outside()

    @property
    def cells(self):
        r = []
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                r.append([j, i, cell])
        return r

    @property
    def outside(self):
        return [cell for cell in self.cells if cell[2] == 1]

    @property
    def d_outside(self):
        return np.array([[x * (0.02 / w), y * (0.015 / h), v] for x, y, v in self.cells if v == 1])

    @property
    def inside(self):
        return np.array([cell for cell in self.cells if cell[2] == 2])

    @property
    def d_inside(self):
        return np.array([[x * (0.02 / w), y * (0.015 / h), v] for x, y, v in self.cells if v == 2])

    @property
    def walls(self):
        return np.array([cell for cell in self.cells if cell[2] == 3])

    @property
    def d_walls(self):
        return np.array([[x * (0.02 / w), y * (0.015 / h), v] for x, y, v in self.cells if v == 3])

    def is_outline(self, cell):
        return self.grid[cell[1]][cell[0]] == 1

    def is_inside(self, cell):
        return self.grid[cell[1]][cell[0]] == 2

    def is_true_outside(self, cell):
        return self.grid[cell[1]][cell[0]] == 3

    def get_coords(self, i, j):
        return i * self.s_w, j * self.s_h

    def __filling(self, i, j):
        neigbs = [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]
        for pixel in neigbs:
            if 0 <= pixel[0] < self.h and 0 <= pixel[1] < self.w:
                if self.grid[pixel[0]][pixel[1]] == 0:
                    self.grid[pixel[0]][pixel[1]] = 2
                    self.__filling(pixel[0], pixel[1])

    def __filling_outside(self):
        for (i, j, _) in self.outside:
            neigbs = [(i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1)]
            for pixel in neigbs:
                if 0 <= pixel[1] < self.h and 0 <= pixel[0] < self.w and self.is_inside(pixel):
                    self.grid[j][i] = 3
                    break

    def plot(self, show_outline=False):
        for cell in self.cells:
            if self.is_outline(cell) and show_outline:
                plt.gca().add_patch(
                    plt.Rectangle(self.get_coords(cell[0], cell[1]), self.s_w, self.s_h, fill=True, color='b'))
            if self.is_inside(cell):
                plt.gca().add_patch(
                    plt.Rectangle(self.get_coords(cell[0], cell[1]), self.s_w, self.s_h, fill=True, color='r'))
            elif self.is_true_outside(cell):
                plt.gca().add_patch(
                    plt.Rectangle(self.get_coords(cell[0], cell[1]), self.s_w, self.s_h, fill=True, color='g'))
            else:
                plt.gca().add_patch(
                    plt.Rectangle(self.get_coords(cell[0], cell[1]), self.s_w, self.s_h, fill=False, lw=0))
        plt.title("Points du contour")
        plt.axis('equal')
        plt.show()
