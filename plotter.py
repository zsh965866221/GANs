# coding = utf-8
from time import sleep

import numpy as np


class LinePlotter:
    def __init__(self, viz):
        self.viz = viz
        self.plots = {}

    def plot(self, var, split, title, x, y):
        if var not in self.plots:
            self.plots[var] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), opts=dict(
                legend=[split],
                title=title,
                xlabel='Epochs',
                ylabel=var
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), win=self.plots[var], name=split, update='append')


