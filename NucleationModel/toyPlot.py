import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np


class ToyPlot(object):
    def __init__(self, x_range, y_range):
        range_x = np.linspace(x_range[0], x_range[1], 100)
        range_y = np.linspace(y_range[0], y_range[1], 100)
        self.extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        self.X, self.Y = np.meshgrid(range_x, range_y)
        pylab.rcParams['figure.figsize'] = 5, 5
        self.repcolordict = {0: 'k-', 1: 'r-', 2: 'g-', 3: 'b-', 4: 'r-'}
        self.contour_range = np.arange(0.0, 1.5, 0.1)
        self._states = None
        self._pes = None
        self._interfaces = None
        self._initcond = None

    def add_pes(self, pes):
        if self._pes is None:
            self._pes = np.vectorize(CallablePES(pes))(self.X, self.Y)

    def plot(self, trajectories=[]):
        fig, ax = plt.subplots()
        if self._pes is not None:
            plt.contour(
                self.X, self.Y, self._pes, levels=self.contour_range,
                colors='k')
        for traj in trajectories:
            plt.plot(
                traj.xyz[:, 0, 0], traj.xyz[:, 0, 1],
                self.repcolordict[trajectories.index(traj) % 5], zorder=2)
        return fig

    def reset(self):
        self._pes = None
        self._interfaces = None
        self._initcond = None
        self._states = None


class CallablePES(object):
    def __init__(self, pes):
        self.pes = pes

    def __call__(self, x, y):
        self.positions = [x, y]
        return self.pes.V(self)