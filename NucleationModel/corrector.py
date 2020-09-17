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
    def correct_1D_point(self, grid_snapshots, x_int):
        snapshots_per_x = {}
        for snapshot in grid_snapshots:
            # Append to key associated with this snapshots value for x
            try:
                snapshots_per_x[snapshot[x_int]].append(tuple(snapshot))
            except Exception:
                snapshots_per_x[snapshot[x_int]] = [tuple(snapshot)]
        means_per_x = {}
        for x in snapshots_per_x:
            means_per_x[x] = self.get_means_from_tuples(snapshots_per_x[x])
        return means_per_x

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
    def correct_2D_point(self, grid_snapshots, x_int, y_int):
        snapshots_per_xy = {}
        for snapshot in grid_snapshots:
            # Append to key associated with this snapshots values for x and y
            try:
                snapshots_per_xy[(snapshot[x_int], snapshot[y_int])] \
                    .append(tuple(snapshot))
            except Exception:
                snapshots_per_xy[(snapshot[x_int], snapshot[y_int])] \
                    = [tuple(snapshot)]
        means_per_xy = {}
        for xy in snapshots_per_xy:
            means_per_xy[xy] = self.get_means_from_tuples(snapshots_per_xy[xy])
        return means_per_xy

    def get_means_from_tuples(tuples):
        return np.mean(list(zip(*tuples)), axis=1)

    def get_modes_from_tuples(tuples):
        return stats.mode(tuples)[0][0]