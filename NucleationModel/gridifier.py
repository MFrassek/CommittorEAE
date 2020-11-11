import numpy as np


class Gridifier():
    def __init__(self, base_snapshots, resolution):
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
        return self.convert_array_to_int_array(
            self.broadcast_round_to_closest_integer(
                self.broadcast_rescale_to_resolution_range(
                    self.broadcast_shift_values_to_start_at_minimum(
                        snapshots))))

    def broadcast_shift_values_to_start_at_minimum(self, snapshots):
        return snapshots - self.minima

    def broadcast_rescale_to_resolution_range(self, snapshots):
        return snapshots * self._inverse_spans_times_resolution

    def broadcast_round_to_closest_integer(self, snapshots):
        return np.floor(snapshots + 0.5)

    def convert_array_to_int_array(self, snapshots):
        return snapshots.astype(int)
