import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
from helperFunctions import function_to_str, flatten_list_of_lists
from autoEncoder import AutoEncoder


def plot_super_map(
        used_variable_names: list,
        name_to_list_position: dict,
        lower_bound,
        upper_bound,
        const,
        pre_stamp,
        method,
        norm="Log",
        **kwargs):
    """
    params:
        used_variable_names: list
            all values to be visited as x_pos
        used_variable_names: list
            all values to be visited as y_pos
    """
    method_name = function_to_str(method)
    if "given" in method_name:
        out_size = 1
        axis_label = "${}$"
    elif "generated" in method_name:
        model = kwargs["model"]
        out_size = model.layers[-1].output_shape[1]
        if model.input_names[0] == "encoded_snapshots":
            "b{}"
        else:
            axis_label = "${}$"

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
                    **kwargs)
                super_map[-1][-1].append(label_map)
    for k in range(out_size):
        print(k)
        fig, axs = plt.subplots(
            len(used_variable_names), len(used_variable_names),
            figsize=(
                const.subfig_size * len(used_variable_names),
                const.subfig_size * len(used_variable_names)))
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
                                np.maximum(
                                    super_map[i][j][0][k][::-1],
                                    const.logvmin / 2),
                                cmap=const.cmap,
                                interpolation='nearest',
                                norm=mpl.colors.LogNorm(
                                    vmin=const.logvmin,
                                    vmax=1-const.logvmin),
                                extent=[0, 1, 0, 1])
                        else:
                            im = new_axs.imshow(
                                np.maximum(
                                    super_map[i][j][0][k][::-1],
                                    const.logvmin / 2),
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
                            np.maximum(
                                super_map[i][j][0][k][::-1],
                                const.logvmin / 2),
                            cmap="seismic",
                            interpolation='nearest',
                            norm=mpl.colors.Normalize(
                                vmin=const.min_label,
                                vmax=const.max_label),
                            extent=[0, 1, 0, 1])
                    # Only sets the leftmost and lowest label.
                    if i == len(used_variable_names) - 1:
                        new_axs.set_xlabel(
                            axis_label.format(used_variable_names[j]),
                            fontsize=const.subfig_size * 10)
                        new_axs.tick_params(
                            labelbottom=True,
                            bottom=True,)
                        new_axs.set_xticks(np.linspace(0, 1, 3))
                        new_axs.set_xticklabels(
                            np.around(
                                np.linspace(
                                    lower_bound[j], upper_bound[j], 3),
                                2),
                            rotation=60,
                            fontsize=const.subfig_size*6)
                    if j == 0:
                        new_axs.set_ylabel(
                            axis_label.format(used_variable_names[i]),
                            fontsize=const.subfig_size * 10)
                        new_axs.tick_params(
                            labelleft=True,
                            left=True)
                        new_axs.set_yticks(np.linspace(0, 1, 3))
                        new_axs.set_yticklabels(np.around(
                            np.linspace(
                                lower_bound[i], upper_bound[i], 3),
                            2),
                            fontsize=const.subfig_size*6)
                else:
                    # Remove all subplots where i >= j.
                    new_axs.axis("off")
        fig.align_labels()
        cax, kw = mpl.colorbar.make_axes([ax for ax in axs])
        cbar = plt.colorbar(im, cax=cax, **kw, extend="both")
        cbar.ax.tick_params(labelsize=const.subfig_size
                            * len(used_variable_names))
        if function_to_str(method).split("_")[-1][:3] == "gen":
            if "partial" in function_to_str(method):
                method_stamp = "genP"
            else:
                method_stamp = "gen"
            plt.savefig("results/{}_{}_{}_{}_outN{}_r{}_map.png"
                        .format(
                            pre_stamp,
                            method_stamp,
                            const.data_stamp,
                            const.model_stamp,
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


def calc_map_given(
        x_pos,
        y_pos,
        resolution,
        grid_snapshots,
        labels,
        weights,
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


def calc_partial_map_given(
        x_pos,
        y_pos,
        resolution,
        grid_snapshots,
        labels,
        weights,
        points_of_interest,
        fill_val=0):
    xys = list(set([(int(ele[x_pos]), int(ele[y_pos]))
                    for ele in points_of_interest]))
    label_map = calc_map_given(
            x_pos=x_pos,
            y_pos=y_pos,
            resolution=resolution,
            grid_snapshots=grid_snapshots,
            labels=labels,
            weights=weights,
            fill_val=fill_val)
    partial_out_map = [[label_map[0][x][y]
                       if (x, y) in xys else float("NaN")
                       for y in range(resolution)]
                       for x in range(resolution)]
    return np.array([partial_out_map])


def calc_map_generated(
        x_pos,
        y_pos,
        resolution,
        minima,
        maxima,
        model,
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
            prediction = calc_map_point(
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


def calc_partial_map_generated(
        x_pos,
        y_pos,
        resolution,
        minima,
        maxima,
        model,
        points_of_interest,
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
        prediction = calc_map_point(
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


def calc_represented_map_generated(
        x_pos,
        y_pos,
        resolution,
        minima,
        maxima,
        model,
        representations):
    assert x_pos != y_pos, "x_pos and y_pos need to differ"
    out_size = model.layers[-1].output_shape[1]
    xy_representations = representations[(x_pos, y_pos)]
    out_map = [[[float("NaN") for i in range(resolution)]
               for j in range(resolution)]
               for k in range(out_size)]
    span_inv_resolution = (maxima - minima) / (resolution - 1)
    norm_xy_representations = \
        (xy_representations * span_inv_resolution) + minima
    for i, norm_representation in enumerate(norm_xy_representations):
        prediction = model.predict([[norm_representation]])[0]
        for j in range(out_size):
            # Take x and y positions from the original xy_representation
            # to assort the prediction to the right grid point
            out_map[j][int(xy_representations[i][x_pos])]\
                [int(xy_representations[i][y_pos])] = prediction[j]
    return np.array(out_map)


def plot_super_scatter(
        used_variable_names: list,
        name_to_list_position: dict,
        lower_bound,
        upper_bound,
        const,
        pre_stamp,
        method,
        minima,
        maxima,
        max_row_len=6,
        **kwargs):
    """Generates a superfigure of scater plots.
    Iterates over the different dimensions and based on
    different input values for one dimensions
    as well as a fixed value fr all other dimensions,
    predicts the reconstructed value for that dimension.
    An optimal encoding and decoding will yield a diagonal
    line for each dimension indifferent of the value
    chosen for the other dimensions.
    """
    row_cnt = ((len(used_variable_names)-1)//max_row_len)+1
    fig, axs = plt.subplots(
        row_cnt, max_row_len,
        figsize=(
            const.subfig_size*max_row_len,
            const.subfig_size*row_cnt*1.3))
    fig.align_labels()

    for i, var_name in enumerate(used_variable_names):
        xs, ys = method(
            x_pos=name_to_list_position[var_name],
            resolution=const.resolution,
            minima=minima,
            maxima=maxima,
            **kwargs)
        if row_cnt > 1:
            new_axs = axs[i//max_row_len]
        else:
            new_axs = axs
        new_axs[i % max_row_len]\
            .scatter(xs, ys, s=const.subfig_size*20)
        new_axs[i % max_row_len]\
            .set_xlim(
                [minima[name_to_list_position[var_name]],
                 maxima[name_to_list_position[var_name]]])
        new_axs[i % max_row_len]\
            .set_ylim(
                [minima[name_to_list_position[var_name]],
                 maxima[name_to_list_position[var_name]]])
        new_axs[i % max_row_len]\
            .set_xlabel(
                "${}$".format(var_name),
                fontsize=const.subfig_size*10)
        new_axs[i % max_row_len].tick_params(
            axis='both',
            which='both',
            top=False,
            labelbottom=True,
            bottom=True,
            left=False,
            labelleft=False)
        new_axs[i % max_row_len].set_xticks(
            np.linspace(min(xs), max(xs), 3))
        new_axs[i % max_row_len].set_xticklabels(
            np.around(
                np.linspace(
                    lower_bound[i],
                    upper_bound[i],
                    3),
                2),
            rotation=60,
            fontsize=const.subfig_size*6)
    # if not all rows are filled
    # remove the remaining empty subplots in the last row
    if len(used_variable_names) % max_row_len != 0:
        for i in range(len(used_variable_names)
                       % max_row_len, max_row_len):
            new_axs[i].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig("results/{}_{}_{}_r{}_scat.png"
                .format(
                    pre_stamp,
                    const.data_stamp,
                    const.model_stamp,
                    const.resolution))
    plt.show()
    return


def calc_scatter_generated(
        x_pos,
        resolution,
        model,
        minima,
        maxima,
        fill_val=0):
    in_size = model.layers[0].output_shape[0][1]
    xs = np.linspace(minima[x_pos], maxima[x_pos], resolution)
    ys = []
    for x in xs:
        prediction = model.predict([[x if x_pos == pos_nr else fill_val
                                    for pos_nr in range(in_size)]])[0]
        ys.append(prediction[x_pos])
    return xs, ys


def calc_represented_scatter_generated(
        x_pos,
        resolution,
        model,
        minima,
        maxima,
        representations):
    x_representations = representations[x_pos]
    xs = np.linspace(minima[x_pos], maxima[x_pos], resolution)
    ys = [float("NaN") for i in range(resolution)]
    span_inv_resolution = (maxima - minima) / (resolution - 1)
    norm_x_representations = \
        (x_representations * span_inv_resolution) + minima
    for i, norm_representation in enumerate(norm_x_representations):
        prediction = model.predict([[norm_representation]])[0]
        ys[int(x_representations[i][x_pos])] = prediction[x_pos]
    return np.array(xs), np.array(ys)


def plot_loss_history(history, file_name):
    plt.figure(figsize=(8, 8))
    for key, log_loss in history.history.items():
        plt.plot(range(1, 1+len(log_loss)), log_loss, label=key)
        plt.scatter(range(1, 1+len(log_loss)), log_loss, s=10)
    plt.ylim(0,)
    plt.xlim(1, len(log_loss))
    plt.xticks(range(1, 1+len(log_loss)))
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.show()


def plot_ground_truth(
        reduced_list_var_names, reduced_name_to_list_position, pipeline,
        const, grid_snapshots, labels, weights, pre_stamp, norm="log"):
    plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp=pre_stamp,
        method=calc_map_given,
        grid_snapshots=grid_snapshots,
        labels=labels,
        weights=weights,
        norm=norm)


def plot_with_different_settings(
        reduced_list_var_names,
        reduced_name_to_list_position,
        const,
        train_ds, val_ds, loss_function,
        pipeline, pre_stamp, minima, maxima):
    autoencoder, autoencoder_1, autoencoder_2, \
        encoder, decoder_1, decoder_2 = \
        AutoEncoder.make_models(len(reduced_list_var_names), loss_function, const)
    history = autoencoder.fit(
        x=train_ds,
        epochs=const.epochs,
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3)])
    with open("results/{}_LossLog_{}_{}.txt".format(
            pre_stamp, const.data_stamp, const.model_stamp), "w") as output:
        output.write("\t".join([key for key in history.history.keys()]))
        output.write("\n")
        for epoch_log in list(zip(*history.history.values())):
            output.write("\t".join(list(map(
                lambda x: str(round(x, 3)), epoch_log))))
            output.write("\n")
    plot_loss_history(history, "results/{}_LossLog_{}_{}.png".format(
        pre_stamp, const.data_stamp, const.model_stamp))
#    Plotter.plot_super_map(
#        used_variable_names = reduced_list_var_names,
#        name_to_list_position = reduced_name_to_list_position,
#        lower_bound=pipeline.lower_bound,
#        upper_bound=pipeline.upper_bound,
#        const = c,
#        pre_stamp = pre_stamp,
#        method = Plotter.calc_partial_map_generated,
#        model = autoencoder_1,
#        minima = minima,
#        maxima = maxima,
#        points_of_interest = train_trimmed_pB_dict)
    plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp=pre_stamp,
        method=calc_map_generated,
        model=autoencoder_1,
        minima=minima,
        maxima=maxima)
    plot_super_scatter(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp=pre_stamp,
        model=autoencoder_2,
        minima=minima,
        maxima=maxima,
        fill_val=0,
        max_row_len=6)


def plot_encoder_decoder(
        const,
        loss_function,
        reduced_list_var_names,
        reduced_name_to_list_position,
        train_ds,
        val_ds,
        pipeline):
    autoencoder, autoencoder_1, autoencoder_2, \
        encoder, decoder_1, decoder_2 = \
        AutoEncoder.make_models(
            len(reduced_list_var_names), loss_function, const)
    autoencoder.fit(
        x=train_ds,
        epochs=const.epochs,
        validation_data=val_ds,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3)])
    plot_encoder(
        reduced_list_var_names=reduced_list_var_names,
        reduced_name_to_list_position=reduced_name_to_list_position,
        pipeline=pipeline,
        const=const,
        loss_function=loss_function,
        encoder=encoder)
    plot_decoder(
        const=const,
        decoder_1=decoder_1)


def plot_encoder(
        reduced_list_var_names,
        reduced_name_to_list_position,
        pipeline,
        const,
        loss_function,
        encoder):
    plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp="EncoderTest"+"_{}".format(function_to_str(loss_function)),
        method=calc_map_generated,
        model=encoder,
        minima=pipeline.minima,
        maxima=pipeline.maxima)


def plot_decoder(
        const, decoder_1):
    x_int = 0
    y_int = 1
    minima = [-10, -10]
    maxima = [10, 10]
    fig, ax = plt.subplots(1, 1)
    plt.imshow(
        np.maximum(
            np.transpose(
                calc_map_generated(
                    x_pos=x_int,
                    y_pos=y_int,
                    minima=minima,
                    maxima=maxima,
                    resolution=const.resolution,
                    model=decoder_1,
                    fill_val=1)[0])[::-1],
            const.logvmin / 2),
        cmap=const.cmap,
        interpolation='nearest',
        norm=mpl.colors.LogNorm(
            vmin=const.logvmin,
            vmax=1.0-const.logvmin),
        extent=[0, 1, 0, 1])
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_xticklabels(
        np.around(
            np.linspace(
                minima[x_int],
                maxima[x_int],
                3),
            2),
        rotation=60)
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.set_yticklabels(np.around(
        np.linspace(
            minima[y_int],
            maxima[y_int],
            3),
        2))
    ax.set_xlabel(
        "$bn{}$".format(x_int),
        fontsize=const.subfig_size * 10)
    ax.set_ylabel(
        "$bn{}$".format(y_int),
        fontsize=const.subfig_size * 10)
    plt.colorbar(extend="both")
    plt.tight_layout()
    plt.savefig("results/Decoder_{}.png".format(const.model_stamp))
    plt.show()


def plot_example_paths_on_latent_space(
        get_paths_function,
        const,
        pipeline,
        reduced_list_var_names,
        steps,
        encoder,
        pre_stamp):
    paths, labels = get_paths_function(const=const)
    latent_paths = [make_latent_path_from_path(
            pipeline=pipeline,
            path=path,
            reduced_list_var_names=reduced_list_var_names,
            steps=steps,
            encoder=encoder)
        for path in paths]
    flattened_latent_paths = flatten_list_of_lists(latent_paths)
    latent_minimum = np.amin(np.transpose(flattened_latent_paths)[0], axis=0)
    latent_maximum = np.amax(np.transpose(flattened_latent_paths)[0], axis=0)
    plot_latent_paths(
        latent_paths=latent_paths,
        labels=labels,
        steps=steps,
        pre_stamp=pre_stamp,
        const=const)
    return latent_minimum, latent_maximum


def make_latent_path_from_path(
        pipeline, path, reduced_list_var_names, steps, encoder):
    bn_size = encoder.layers[-1].output_shape[1]
    bn_path = pipeline.bound_normalize(path)
    bnr_path = pipeline.reduce(bn_path, reduced_list_var_names)
    path_len = len(bnr_path)
    latent_path = [encoder.predict([[bnr_path[int(path_len*i/(steps+1))]]])[0]
                   for i in range(steps+1)]
    if bn_size == 1:
        return [[path[0], i] for i, path in enumerate(latent_path)]
    elif bn_size == 2:
        return latent_path
    else:
        raise ValueError(
            "Data of dimensionality {} cannot be plotted".format(bn_size))


def plot_latent_paths(latent_paths, labels, steps, pre_stamp, const):
    for plot_path, label in zip(latent_paths, labels):
        plt.plot(*np.transpose(plot_path), label=str(label))
    plt.xlabel("$BN_1$")
    if const.bottleneck_size == 1:
        plt.ylabel(r"Progress along path [%]")
        plt.yticks(
            [steps*i/10 for i in range(11)],
            [100*i/10 for i in range(11)])
        plt.ylim(0, steps)
    else:
        plt.ylabel("$BN_2$")
    plt.title("Paths mapped onto the latent space")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.subplots_adjust(right=0.82)
    plt.savefig("results/{}_LatentSpacePath_plot_{}_{}D.png".format(
        pre_stamp, const.model_stamp, const.bottleneck_size))
    plt.show()


def plot_relative_importances(names, values):
    dollar_names = ["$"+name+"$" for name in names]
    precentages = [value * 100 for value in values]
    plt.bar(dollar_names, precentages)
    plt.ylabel("Relatve importance [%]")
    plt.ylim(0, 100)
    plt.xticks(rotation=60)
    plt.savefig("results/LinearComponents.png")
    plt.show()


def plot_single_map(
        x_int, y_int, const,
        pipeline, reduced_list_var_names,
        stamp, method, norm="log", **kwargs):
    fig, ax = plt.subplots(1, 1)
    if norm == "log":
        cmap = const.cmap
        norm = mpl.colors.LogNorm(
            vmin=const.logvmin,
            vmax=1.0-const.logvmin)
    else:
        cmap = "seismic"
        norm = mpl.colors.Normalize(
            vmin=const.min_label,
            vmax=const.max_label)
    plt.imshow(
        np.maximum(
            method(
                x_pos=y_int,
                y_pos=x_int,
                resolution=const.resolution,
                **kwargs)[0][::-1],
            const.logvmin / 2),
        cmap=cmap,
        interpolation='nearest',
        norm=norm,
        extent=[0, 1, 0, 1])
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_xticklabels(
        np.around(
            np.linspace(
                pipeline.lower_bound[x_int],
                pipeline.upper_bound[x_int],
                3),
            2),
        rotation=60)
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.set_yticklabels(np.around(
        np.linspace(
            pipeline.lower_bound[y_int],
            pipeline.upper_bound[y_int],
            3),
        2))
    ax.set_xlabel(
        "${}$".format(reduced_list_var_names[x_int]),
        fontsize=const.subfig_size * 10)
    ax.set_ylabel(
        "${}$".format(reduced_list_var_names[y_int]),
        fontsize=const.subfig_size * 10)
    plt.colorbar(extend="both")
    plt.tight_layout()
    plt.savefig("results/{}_x{}_y_{}.png".format(stamp, x_int, y_int))
    plt.show()


def plot_reconstruction_from_latent_space(
        reduced_list_var_names,
        latent_minimum, latent_maximum,
        steps, recon_decoder, pre_stamp):
    fig = go.Figure()
    var_names = ["$"+name+"$" for name
                 in reduced_list_var_names
                 + [reduced_list_var_names[0]]]
    for i, val in enumerate(np.linspace(np.floor(
            latent_minimum), np.ceil(latent_maximum), steps)):
        prediction = recon_decoder.predict([val])[0]
        prediction = np.append(prediction, prediction[0])
        fig.add_trace(
            go.Scatterpolar(
                r=prediction,
                theta=var_names,
                name="{:.1f}".format(val),
                showlegend=True,
                line=dict(color="rgb({},{},{})".format(
                    0.8 - 0.6 * i / steps,
                    0.2 + 0.6 * i / steps,
                    0.2 + 0.6 * i / steps))))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, tickangle=0),
            angularaxis=dict(tickfont=dict(size=18))),
        title_text="Path reconstruction from latent space", title_x=0.5,
        legend_title_text="$\ BN_1 input$")
    fig.write_image("results/{}_PathReconstruction.png".format(pre_stamp))
    fig.show()
