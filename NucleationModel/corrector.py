import numpy as np


class Corrector():
    @classmethod
    def correct_1D_row(self, grid_snapshots):
        all_representations = {}
        for x in range(len(grid_snapshots[0])):
            x_representation = self.correct_1D_point(
                grid_snapshots=grid_snapshots,
                x_int=x)
            x_representation = np.array(
                [value for value in x_representation.values()])
            all_representations[x] = x_representation
        return all_representations

    @classmethod
    def correct_1D_point(self, grid_snapshots, **kwargs):
        snapshots_per_x = {}
        self.fill_dictionary(snapshots_per_x, grid_snapshots, **kwargs)
        means_per_x = self.get_position_means_dictionary(snapshots_per_x)
        return means_per_x

    @classmethod
    def fill_dictionary(self, dictionary, grid_snapshots, **kwargs):
        for snapshot in grid_snapshots:
            key_tuple = tuple(snapshot[pos_int] for pos_int in kwargs.values())
            self.attempt_adding_snapshot_to_dictionary_at_key(
                dictionary, snapshot, key_tuple)

    @classmethod
    def attempt_adding_snapshot_to_dictionary_at_key(
            self, dictionary, snapshot, key):
        try:
            dictionary[key].append(tuple(snapshot))
        except Exception:
            dictionary[key] = [tuple(snapshot)]

    @classmethod
    def get_position_means_dictionary(
            self, snapshots_per_position_dictionary):
        means_per_position = {}
        for key, value in snapshots_per_position_dictionary.items():
            means_per_position[key] = self.get_means_from_tuples(value)
        return means_per_position

    @classmethod
    def correct_2D_grid(self, grid_snapshots):
        all_representations = {}
        for x in range(len(grid_snapshots[0])):
            for y in range(0, x):
                xy_representation = self.correct_2D_point(
                    grid_snapshots=grid_snapshots,
                    x_int=x,
                    y_int=y)
                xy_representation = np.array(
                    [value for value in xy_representation.values()])
                all_representations[(x, y)] = xy_representation
                all_representations[(y, x)] = xy_representation
        return all_representations

    @classmethod
    def correct_2D_point(self, grid_snapshots, **kwargs):
        snapshots_per_xy = {}
        self.fill_dictionary(snapshots_per_xy, grid_snapshots, **kwargs)
        means_per_xy = self.get_position_means_dictionary(snapshots_per_xy)
        return means_per_xy

    def get_means_from_tuples(tuples):
        return np.mean(list(zip(*tuples)), axis=1)
