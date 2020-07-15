import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helperFunctions import function_to_str


class Plotter():
    @staticmethod
    def plot_super_map(
            used_variable_names: list,
            name_to_list_position: dict,
            lower_bound,
            upper_bound,
            const,
            pre_stamp,
            method,
            model=None,
            minima=None,
            maxima=None,
            grid_snapshots=None,
            labels=None,
            weights=None,
            points_of_interest=None,
            fill_val=0,
            norm="Log"):
        """
        params:
            used_variable_names: list
                all values to be visited as x_pos
            used_variable_names: list
                all values to be visited as y_pos
            model:
                tf model used for predictions
                default None
            fill_val: float/int
                value assigned to all dimensions not specifically targeted
                default 0 (mean of the normalized list)
        """
        if model is None:
            suptitle = "Given labels depending on input"
            model_name = "Given"
            out_size = 1
        else:
            suptitle = "Predicted {} depending on {}"\
                        .format(model.output_names[0], model.input_names[0])
            model_name = model.name
            out_size = model.layers[-1].output_shape[1]

        super_map = []
        for i, var_name_i in enumerate(used_variable_names):
            super_map.append([])
            for j, var_name_j in enumerate(used_variable_names):
                super_map[-1].append([])
                if j < i:
                    print("{}: {}\t{}: {}".format(
                        i, var_name_i,
                        j, var_name_j))
                    label_map = method(
                        x_pos=name_to_list_position[var_name_i],
                        y_pos=name_to_list_position[var_name_j],
                        resolution=const.resolution,
                        minima=minima,
                        maxima=maxima,
                        grid_snapshots=grid_snapshots,
                        labels=labels,
                        weights=weights,
                        model=model,
                        points_of_interest=points_of_interest,
                        fill_val=fill_val)
                    super_map[-1][-1].append(label_map)
        for k in range(out_size):
            print(k)
            fig, axs = plt.subplots(
                len(used_variable_names), len(used_variable_names),
                figsize=(
                    const.subfig_size * len(used_variable_names),
                    const.subfig_size * len(used_variable_names)))
            fig.suptitle(
                suptitle,
                fontsize=const.subfig_size*len(used_variable_names)*2)

            for i, _ in enumerate(used_variable_names):
                for j, _ in enumerate(used_variable_names):
                    # Defines new_axs to take care of different
                    # handling of only one column of subplots.
                    if len(used_variable_names) == 1:
                        new_axs = axs[i]
                    else:
                        new_axs = axs[i][j]
                    new_axs.tick_params(
                        axis='both',
                        which='both',
                        top=False,
                        labelleft=False,
                        left=False,
                        labelbottom=False,
                        bottom=False)
                    if j < i:
                        if norm == "Log":
                            if const.min_label >= 0:
                                im = new_axs.imshow(
                                    super_map[i][j][0][k][::-1],
                                    cmap=const.cmap,
                                    interpolation='nearest',
                                    norm=mpl.colors.LogNorm(
                                        vmin=const.logvmin,
                                        vmax=1-const.logvmin),
                                    extent=[0, 1, 0, 1])
                            else:
                                im = new_axs.imshow(
                                    super_map[i][j][0][k][::-1],
                                    cmap=const.cmap,
                                    interpolation='nearest',
                                    norm=mpl.colors.SymLogNorm(
                                        linthresh=0.01*(const.max_label
                                                        - const.min_label),
                                        linscale=0.1*(const.max_label
                                                      - const.min_label),
                                        vmin=const.min_label,
                                        vmax=const.max_label),
                                    extent=[0, 1, 0, 1])
                        else:
                            im = new_axs.imshow(
                                super_map[i][j][0][k][::-1],
                                cmap=const.cmap,
                                interpolation='nearest',
                                norm=mpl.colors.Normalize(
                                    vmin=const.min_label,
                                    vmax=const.max_label),
                                extent=[0, 1, 0, 1])
                        # Only sets the leftmost and lowest label.
                        if i == len(used_variable_names) - 1:
                            new_axs.set_xlabel(
                                "${}$".format(used_variable_names[j]),
                                fontsize=const.subfig_size * 10)
                            new_axs.tick_params(
                                labelbottom=True,
                                bottom=True)
                            new_axs.set_xticks(np.linspace(0, 1, 3))
                            new_axs.set_xticklabels(
                                np.around(
                                    np.linspace(
                                        lower_bound[j],
                                        upper_bound[j],
                                        3),
                                    2),
                                rotation=60,
                                fontsize=const.subfig_size*6)
                        if j == 0:
                            new_axs.set_ylabel(
                                "${}$".format(used_variable_names[i]),
                                fontsize=const.subfig_size * 10)
                            new_axs.tick_params(
                                labelleft=True,
                                left=True)
                            new_axs.set_yticks(np.linspace(0, 1, 3))
                            new_axs.set_yticklabels(np.around(
                                np.linspace(
                                    lower_bound[i],
                                    upper_bound[i],
                                    3),
                                2),
                                fontsize=const.subfig_size*6)
                        # Overwrites labels if predictions are based on the bn.
                        if model is not None:
                            if model.input_names[0] == "encoded_snapshots":
                                if i == len(used_variable_names) - 1:
                                    new_axs.set_xlabel(
                                        "b{}".format(used_variable_names[j]),
                                        fontsize=const.subfig_size * 10)
                                if j == 0:
                                    new_axs.set_ylabel(
                                        "b{}".format(used_variable_names[i]),
                                        fontsize=const.subfig_size * 10)
                    else:
                        # Remove all subplots where i >= j.
                        new_axs.axis("off")
            cax, kw = mpl.colorbar.make_axes([ax for ax in axs])
            cbar = plt.colorbar(im, cax=cax, **kw, extend="both")
            cbar.ax.tick_params(labelsize=const.subfig_size
                                * len(used_variable_names))
            if function_to_str(method).split("_")[-1][:3] == "gen":
                if "partial" in function_to_str(method):
                    method_stamp = "genP"
                else:
                    method_stamp = "gen"
                plt.savefig("results/{}_{}_{}_{}_fv{}_outN{}_r{}_map.png"
                            .format(
                                pre_stamp,
                                method_stamp,
                                const.data_stamp,
                                const.model_stamp,
                                fill_val,
                                k,
                                const.resolution))
            elif function_to_str(method).split("_")[-1][:3] == "giv":
                if "partial" in function_to_str(method):
                    method_stamp = "givP"
                else:
                    method_stamp = "giv"
                plt.savefig("results/{}_{}_{}_r{}_map.png"
                            .format(
                                pre_stamp,
                                method_stamp,
                                const.data_stamp,
                                const.resolution))
            plt.show()
        return super_map

    @staticmethod
    def calc_map_given(
            x_pos,
            y_pos,
            resolution,
            minima=None,
            maxima=None,
            grid_snapshots=None,
            labels=None,
            weights=None,
            model=None,
            points_of_interest=None,
            fill_val=0):
        label_map = [[0 for y in range(resolution)] for x in range(resolution)]
        weight_map = [[0 for y in range(resolution)]
                      for x in range(resolution)]
        for nr, snapshot in enumerate(grid_snapshots):
            x_int = int(snapshot[x_pos])
            y_int = int(snapshot[y_pos])
            if x_int >= 0 and x_int <= resolution-1 and y_int >= 0 \
                    and y_int <= resolution-1:
                label_map[x_int][y_int] = label_map[x_int][y_int] \
                    + labels[nr] \
                    * weights[nr]
                weight_map[x_int][y_int] = weight_map[x_int][y_int] \
                    + weights[nr]
        label_map = [[label_map[i][j] / weight_map[i][j]
                     if weight_map[i][j] > 0 else float("NaN")
                     for j in range(len(label_map[i]))]
                     for i in range(len(label_map))]
        return np.array([label_map])

    @staticmethod
    def calc_partial_map_given(
            x_pos,
            y_pos,
            resolution,
            minima=None,
            maxima=None,
            grid_snapshots=None,
            labels=None,
            weights=None,
            model=None,
            points_of_interest=None,
            fill_val=0):
        xys = list(set([(int(ele[x_pos]), int(ele[y_pos]))
                        for ele in points_of_interest]))
        label_map = Plotter.calc_map_given(
                x_pos=x_pos,
                y_pos=y_pos,
                resolution=resolution,
                minima=minima,
                maxima=maxima,
                grid_snapshots=grid_snapshots,
                labels=labels,
                weights=weights,
                model=model,
                points_of_interest=points_of_interest,
                fill_val=fill_val)
        partial_out_map = [[label_map[0][x][y]
                           if (x, y) in xys else float("NaN")
                           for y in range(resolution)]
                           for x in range(resolution)]
        return np.array([partial_out_map])

    @staticmethod
    def calc_map_generated(
            x_pos,
            y_pos,
            resolution,
            minima=None,
            maxima=None,
            grid_snapshots=None,
            labels=None,
            weights=None,
            model=None,
            points_of_interest=None,
            fill_val=0):
        """
        Makes predictions over the full (normalized) range of
        two input variables with all other variables fixed to a specific value.
        Outputs the predictions in the form of a list of lists for plotting.
        params:
            model:
                tf model used for predictions
            resolution: int
                resolution of the produced output list and corresponding figure
            x_pos: int
                index of the variable projected on the x-axis
            y_pos: int
                index of the variable projected on the y-axis
            fill_val: float/int
                value assigned to all dimensions not specifically targeted.
                default 0 (mean of the normalized list)
        """
        assert x_pos != y_pos, "x_pos and y_pos need to differ"
        in_size = model.layers[0].output_shape[0][1]
        out_size = model.layers[-1].output_shape[1]
        xs = np.linspace(minima[x_pos], maxima[x_pos], resolution)
        ys = np.linspace(minima[y_pos], maxima[y_pos], resolution)
        out_map = [[] for i in range(out_size)]
        for x in xs:
            out_current_row = [[] for i in range(out_size)]
            for y in ys:
                # make predicition for current grid point
                prediction = Plotter.calc_map_point(
                    model=model,
                    x=x,
                    y=y,
                    x_pos=x_pos,
                    y_pos=y_pos,
                    in_size=in_size,
                    fill_val=fill_val)
                for i in range(out_size):
                    out_current_row[i].append(prediction[i])
            for i in range(out_size):
                out_map[i].append(out_current_row[i])
        return np.array(out_map)

    @staticmethod
    def calc_partial_map_generated(
            x_pos,
            y_pos,
            resolution,
            minima=None,
            maxima=None,
            grid_snapshots=None,
            labels=None,
            weights=None,
            model=None,
            points_of_interest=None,
            fill_val=0):
        assert x_pos != y_pos, "x_pos and y_pos need to differ"
        in_size = model.layers[0].output_shape[0][1]
        out_size = model.layers[-1].output_shape[1]
        xys = list(set([(int(ele[x_pos]), int(ele[y_pos]))
                       for ele in points_of_interest]))
        out_map = [[[float("NaN") for i in range(resolution)]
                   for j in range(resolution)]
                   for k in range(out_size)]
        xs = np.linspace(minima[x_pos], maxima[x_pos], resolution)
        ys = np.linspace(minima[y_pos], maxima[y_pos], resolution)
        for x, y in xys:
            prediction = Plotter.calc_map_point(
                model=model,
                x=xs[x],
                y=ys[y],
                x_pos=x_pos,
                y_pos=y_pos,
                in_size=in_size,
                fill_val=fill_val)
            for i in range(out_size):
                out_map[i][x][y] = prediction[i]
        return np.array(out_map)

    @staticmethod
    def calc_map_point(
            model,
            x,
            y,
            x_pos,
            y_pos,
            in_size,
            fill_val=0):
        return model.predict([[x if x_pos == pos_nr else y
                              if y_pos == pos_nr else fill_val
                              for pos_nr in range(in_size)]])[0]

    @staticmethod
    def plot_super_scatter(
            used_variable_names: list,
            name_to_list_position: dict,
            const,
            pre_stamp,
            model,
            minima,
            maxima,
            fill_val=0,
            max_row_len=6):
        """Generates a superfigure of scater plots.
        Iterates over the different dimensions and based on
        different input values for one dimensions
        as well as a fixed value fr all other dimensions,
        predicts the reconstructed value for that dimension.
        An optimal encoding and decoding will yield a diagonal
        line for each dimension indifferent of the value
        chosen for the other dimensions.
        """
        suptitle = "Predicted snapshots depending on input"
        row_cnt = ((len(used_variable_names)-1)//max_row_len)+1
        fig, axs = plt.subplots(
            row_cnt, max_row_len,
            figsize=(
                const.subfig_size*max_row_len,
                const.subfig_size*row_cnt*1.3))
        fig.suptitle(
            suptitle,
            fontsize=const.subfig_size*max_row_len*2,
            y=1.04 - 0.04*row_cnt)

        for i in used_variable_names:
            xs, ys = Plotter.calc_scatter_generated(
                model=model,
                minima=minima,
                maxima=maxima,
                x_pos=name_to_list_position[i],
                resolution=const.resolution,
                fill_val=fill_val)
            if row_cnt > 1:
                new_axs = axs[(used_variable_names.index(i))//max_row_len]
            else:
                new_axs = axs
            new_axs[used_variable_names.index(i) % max_row_len].tick_params(
                axis='both',
                which='both',
                top=False,
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False)
            im = new_axs[used_variable_names.index(i) % max_row_len]\
                .scatter(xs, ys, s=const.subfig_size*20)
            new_axs[used_variable_names.index(i) % max_row_len]\
                .set_xlim(
                    [minima[name_to_list_position[i]],
                     maxima[name_to_list_position[i]]])
            new_axs[used_variable_names.index(i) % max_row_len]\
                .set_ylim(
                    [minima[name_to_list_position[i]],
                     maxima[name_to_list_position[i]]])
            new_axs[used_variable_names.index(i) % max_row_len]\
                .set_xlabel(
                    "${}$".format(i),
                    fontsize=const.subfig_size*10)
        # if not all rows are filled
        # remove the remaining empty subplots in the last row
        if len(used_variable_names) % max_row_len != 0:
            for i in range(len(used_variable_names)
                           % max_row_len, max_row_len):
                new_axs[i].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.8])
        plt.savefig("results/{}_{}_{}_fv{}_r{}_scat.png"
                    .format(
                        pre_stamp,
                        const.data_stamp,
                        const.model_stamp,
                        fill_val,
                        const.resolution))
        plt.show()
        return

    @staticmethod
    def calc_scatter_generated(
            model,
            minima,
            maxima,
            x_pos,
            resolution,
            fill_val=0):
        in_size = model.layers[0].output_shape[0][1]
        xs = np.linspace(minima[x_pos], maxima[x_pos], resolution)
        ys = []
        for x in xs:
            prediction = model.predict([[x if x_pos == pos_nr else fill_val
                                        for pos_nr in range(in_size)]])[0]
            ys.append(prediction[x_pos])
        return xs, ys
