# -*- coding: utf-8 -*-
"""
Olfactory search environment represented by a video of smoke released on an open field. Taken from Yanagawa's master thesis
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from src.utils.geometry import Point

class SmokeVideo:
    def __init__(self, Nframes, width, height, srcpos, xspace, yspace, h5data=None):

        # self.h5data = h5data
        # self.px2mm = px2mm
        self.mmsrc = srcpos
        self.xs = xspace
        self.ys = yspace
        self.xr = max(self.xs) - min(self.xs)
        self.yr = max(self.ys) - min(self.ys)
        self.width = width
        self.height = height
        self.extent = tuple(xspace + yspace)
        self.vmin = 0
        self.vmax = 255
        self.colormap = "magma"
        # self.plume = np.zeros((height, width))
        if h5data is not None:
            with h5py.File(h5data, 'r') as hf:
                self.plume = hf['frames'][0:Nframes]

    @property
    def pxsrc(self):

        x = (self.mmsrc[0] - min(self.xs))/(max(self.xs) - min(self.xs))
        y = (self.mmsrc[1] - min(self.ys))/(max(self.ys) - min(self.ys))
        x *= self.width
        y *= self.height

        return Point(x, y)

    def mm2px(self, mm: Point):
        xratio = ((self.width) / self.xr)
        yratio = ((self.height) / self.yr)

        w = int(mm.x * xratio + self.pxsrc.x)
        h = int(mm.y * yratio + self.pxsrc.y)
        #print(mm.x,mm.y,w,h)

        return Point(w, h)

    def sample(self, plume, t, x, y):

        c = plume[t, y, x]
        # c = self.plume[t, y, x]
        # if c > 0: print(f'{c}, {x}, {y}')
        return c

    def sample_at(self, t, p: Point, plume=None):
        if plume is not None:
            c = plume[t, int(p.x), int(p.y)]

        else:
            c = self.plume[t, int(p.x), int(p.y)]
        # if c > 0: print(f'{c}, {int(p.x)}, {int(p.y)}')
        return c

    # def sample_hit(self, t, p, plume=None, threshold=1, noise=False):

    #     if noise:
    #         threshold += np.random.randint(1, 10)

    #     if plume is not None:
    #         return int(self.sample_at(t, p, plume) > threshold)

    #     else:
    #         return int(self.sample_at(t, p) > threshold)

    def hit_at(self, t, p, plume=None, threshold=1, noise=False):

        if noise:
            threshold += np.random.randint(1, 10)
        #print(self.sample_at(t, p, plume))
        if plume is not None:
            return int(self.sample_at(t, p, plume) > threshold)

        else:
            return int(self.sample_at(t, p) > threshold)

    def set_animation(self):

        fig, ax = plt.subplots()

        X, Y = np.mgrid[0:self.height, 0:self.width]
        img = ax.imshow(X,
                        origin='lower',
                        extent=self.extent,
                        vmin=0,
                        vmax=255,
                        cmap=self.colormap)

        line, = ax.plot([], lw=1, c='w')
        fig.canvas.draw()

        plt.show(block=False)

        return fig, img, line
