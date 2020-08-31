from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

from caput import config
from drift.util import util

def fetch_layout(conf):
    ltype = conf['type']
    return LAYOUT_TYPES[ltype].from_config(conf)

class SquareGridLayout(config.Reader):

    spacing = config.Property(proptype=float, default=6.)
    grid_size = config.Property(proptype=int, default=3)

    @util.cache_last
    def __call__(self):
        pos_u, pos_v = np.meshgrid(
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size),
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size))
        return np.column_stack((pos_u.T.flat, pos_v.T.flat))

def rotate_coord(coord,rot):
    x,y = coord
    Rot = np.deg2rad(rot)
    xprime = x*np.cos(Rot)-y*np.sin(Rot)
    yprime = x*np.sin(Rot)+y*np.cos(Rot)
    return [xprime,yprime]

class SquareGridLayoutRotated(config.Reader):

    spacing = config.Property(proptype=float, default=6.)
    grid_size = config.Property(proptype=int, default=3)
    rotation_angle = config.Property(proptype=float, default=0)

    @util.cache_last
    def __call__(self):
        pos_u, pos_v = np.meshgrid(
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size),
            np.linspace(0, self.spacing*(self.grid_size-1), self.grid_size))
        
        temp = np.column_stack((pos_u.T.flat, pos_v.T.flat))

        for i in range(len(temp)):
        	temp[i] = rotate_coord(temp[i],self.rotation_angle)

        return temp


class LayoutFile(config.Reader):

    filename = config.Property(proptype=str)

    @util.cache_last
    def __call__(self):
        pos_u, pos_v = np.loadtxt(self.filename, unpack=True)
        return np.column_stack((pos_u, pos_v))


LAYOUT_TYPES = {
    'file': LayoutFile,
    'square_grid': SquareGridLayout,
    'square_grid_rot': SquareGridLayoutRotated,
}
