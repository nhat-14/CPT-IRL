# -*- coding: utf-8 -*-
"""
Olfactory search environment represented by a frames of gas mappings taken from a wind tunnel with a gas sensor array on its floor
"""
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from utils.geometry import Point
from utils import colorize

class WindTunnel(object):
    def __init__(self, h5data, Nframes, width, height, srcpos, xspace,
                 yspace):

        self.h5data = h5data
        self.mmsrc = srcpos
        self.xs = xspace
        self.ys = yspace
        self.xr = max(self.xs) - min(self.xs)
        self.yr = max(self.ys) - min(self.ys)
        self.width = width
        self.height = height
        self.extent = tuple(xspace + yspace)
        self.vmin = 0.
        self.vmax = 6.32e-04
        self.hit_threshold = 1.5e-04
        # self.vmax = 2.4652e-04
        self.colormap = "plasma"
        # self.plume = np.zeros((height, width))
        with h5py.File(self.h5data, 'r') as hf:
            self.plume = hf['frames'][0:Nframes]

    @property
    def pxsrc(self):

        x = (self.mmsrc[0] - min(self.xs)) / (max(self.xs) - min(self.xs))
        y = (self.mmsrc[1] - min(self.ys)) / (max(self.ys) - min(self.ys))
        x *= self.width
        y *= self.height

        return Point(x, y)

    def mm2px(self, mm: Point):
        xratio = ((self.width) / self.xr)
        yratio = ((self.height) / self.yr)

        w = int(mm.x * xratio)
        h = int(mm.y * yratio)

        return Point(w, h)

    def sample_at(self, t, p: Point, radius=1):

        # get mask containing only points within radius of pos
        # xr, yr = (self.xr, self.yr)
        xgap, ygap = (complex(0, self.width), complex(0, self.height))
        # xgap, ygap = (complex(0, xr), complex(0, yr))
        # XS, YS = np.mgrid[xr[0]:xr[1]:complex(0, self.height), yr[0]:yr[1]:complex(0, self.width)]
        XS, YS = np.mgrid[0:self.width:xgap, 0:self.height:ygap]
        # XS, YS = np.mgrid[0:xr:xgap, 0:yr:ygap]

        dxs = int(np.round(p.x)) - XS
        dys = int(np.round(p.y)) - YS

        mask = (dxs**2 + dys**2 < radius**2)

        mat = self.plume[t]

        # sample = self.plume[t, mask].sum()
        # sample = mat[mask.T].sum()
        sample = mat[p.x, p.y]
        # sample = round(sample * (255/self.vmax))
        # if sample >= self.hit_threshold:

            # print(colorize(f"Odor at ({p.x}, {p.y}): {sample}", "red"))

        return sample

    def hit_at(self, t, p, threshold=1, noise=False):

        if noise:
            threshold += np.random.randint(1, 10)

        return int(self.sample_at(t, p) > self.hit_threshold)

    def set_animation(self):

        fig, ax = plt.subplots()

        X, Y = np.mgrid[0:self.height, 0:self.width]
        img = ax.imshow(X,
                        origin='lower',
                        extent=self.extent,
                        vmin=self.vmin,
                        vmax=self.vmax,
                        cmap=self.colormap)

        ax.add_artist(
            plt.Circle((0, 300), 50, color='yellow', fill=False,
                       linestyle='--'))
        line, = ax.plot([], lw=1, c='w')
        fig.canvas.draw()

        plt.show(block=False)

        return fig, img, line