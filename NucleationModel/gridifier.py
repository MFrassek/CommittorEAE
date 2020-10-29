import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class Gridifier():
    def __init__(self, base_snapshots, resolution):
        self._dimensions = len(base_snapshots[0])
        self._minima = np.amin(base_snapshots, axis=0)
        self._maxima = np.amax(base_snapshots, axis=0)
        self._spans = self._maxima - self._minima
        self._resolution = resolution
        self._inverse_spans_times_resolution = \
            1 / self._spans * (self._resolution - 1)

    @property
    def resolution(self):
        return self._resolution

    @property
    def minima(self):
        return self._minima

    @property
    def maxima(self):
        return self._maxima

    def gridify_snapshots(self, snapshots):
        """Take a list of snapshots and bin all entries to the closest
        gridpoint.
        """
        return self.broadcast_round_to_closest_integer(
            self.broadcast_rescale_to_resolution_range(
                self.broadcast_shift_values_to_start_at_minimum(snapshots)))

    def broadcast_shift_values_to_start_at_minimum(self, snapshots):
        return snapshots - self.minima

    def broadcast_rescale_to_resolution_range(self, snapshots):
        return snapshots * self._inverse_spans_times_resolution

    def broadcast_round_to_closest_integer(self, snapshots):
        return np.floor(snapshots + 0.5)

    def plot_distribution(
            self, grid_snapshots, max_row_len,
            subfig_size, var_names, file_name):

        cols = np.transpose(grid_snapshots)
        dimensions = len(grid_snapshots[0])
        # for col in cols:
        #     plt.plot()
        #     plt.hist(col, self._resolution)

        #     plt.show()
        suptitle = "Distribution of input"
        row_cnt = ((dimensions-1)//max_row_len)+1
        fig, axs = plt.subplots(
            row_cnt, max_row_len,
            figsize=(
                subfig_size*max_row_len,
                subfig_size*row_cnt*1.3))
        fig.suptitle(
            suptitle,
            fontsize=subfig_size*max_row_len*2,
            y=1.04 - 0.04*row_cnt)

        for i in range(dimensions):
            if row_cnt > 1:
                new_axs = axs[i//max_row_len]
            else:
                new_axs = axs
            new_axs[i % max_row_len].tick_params(
                axis='both',
                which='both',
                top=False,
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False)
            im = new_axs[i % max_row_len]\
                .hist(cols[i], self._resolution)
            new_axs[i % max_row_len]\
                .set_xlabel("${}$".format(var_names[i]),
                            fontsize=subfig_size*10)
        # if not all rows are filled
        # remove the remaining empty subplots in the last row
        if dimensions % max_row_len != 0:
            for i in range(dimensions % max_row_len, max_row_len):
                new_axs[i].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.8])
        plt.savefig("hist_{}_{}.png".format(file_name, self._resolution))
        plt.show()
