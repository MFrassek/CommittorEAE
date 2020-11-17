import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
from helperFunctions import *
from autoEncoder import AutoEncoder


def plot_super_map(pipeline, const, pre_stamp, method, **kwargs):
    method_name = function_to_str(method)
    out_size = get_out_size(method_name, **kwargs)
    super_map = calculate_super_map(method, const, **kwargs)
    plot_k_super_maps(
            method_name, const, out_size, super_map, pipeline, pre_stamp)
    return super_map


def get_out_size(method_name, **kwargs):
    if "given" in method_name:
        return 1
    elif "generated" in method_name:
        return kwargs["model"].layers[-1].output_shape[1]


def calculate_super_map(method, const, **kwargs):
    return [[method(
            x_pos=const.used_name_to_list_position[var_name_i],
            y_pos=const.used_name_to_list_position[var_name_j],
            resolution=const.resolution,
            **kwargs) if j < i else []
          for j, var_name_j in enumerate(const.used_variable_names)]
          for i, var_name_i in enumerate(const.used_variable_names)]


def plot_k_super_maps(
        method_name, const, out_size, super_map, pipeline, pre_stamp):
    cmap = select_color_map(method_name, const)
    for k in range(out_size):
        print(k)
        fig, axs = prepare_subplots(const)
        make_subplot_heatmaps(super_map, axs, k, const, cmap)
        make_axes_and_labels(const, axs, pipeline, fig)
        make_color_bar(axs, const)
        plt.savefig(get_output_file_name(method_name, pre_stamp, const, k))
        plt.show()


def prepare_subplots(const):
    dimensions_cnt = len(const.used_variable_names)
    return plt.subplots(dimensions_cnt, dimensions_cnt, figsize=(
        const.subfig_size * dimensions_cnt,
        const.subfig_size * dimensions_cnt))


def select_color_map(method_name, const):
    if "density" not in method_name:
        return const.label_cmap
    else:
        return const.density_cmap


def make_subplot_heatmaps(super_map, axs, k, const, cmap):
    for i, sub_map_row in enumerate(super_map):
        for j, sub_map in enumerate(sub_map_row):
            if j < i:
                axs[i][j].imshow(
                        np.maximum(sub_map[k][::-1], const.logvmin / 2),
                        cmap=cmap,
                        interpolation='nearest',
                        norm=mpl.colors.LogNorm(
                            vmin=const.logvmin, vmax=1 - const.logvmin),
                        extent=[0, 1, 0, 1])


def make_axes_and_labels(const, axs, pipeline, fig):
    remove_all_axis_labels(axs)
    remove_empty_subplot_axes(axs)
    set_x_axis_label_for_lowest_subplots(const, axs, pipeline)
    set_y_axis_label_for_leftmost_subplots(const, axs, pipeline)
    fig.align_labels()


def remove_all_axis_labels(axs):
    for i in range(len(axs)):
        for j in range(len(axs)):
            axs[i][j].tick_params(
                axis='both', which='both', top=False, labelleft=False,
                left=False, labelbottom=False, bottom=False)


def remove_empty_subplot_axes(axs):
    for i in range(len(axs)):
        for j in range(len(axs)):
            if j >= i:
                axs[i][j].axis("off")


def set_x_axis_label_for_lowest_subplots(const, axs, pipeline):
    i = len(const.used_variable_names) - 1
    for j in range(i):
        print(i, j)
        j_name = const.used_variable_names[j]
        pipeline_j_int = const.name_to_list_position[j_name]
        axs[i][j].set_xlabel(
            "${}$".format(j_name), fontsize=const.subfig_size * 10)
        set_xtick_labels(
            axs[i][j], pipeline.lower_bound, pipeline.upper_bound,
            pipeline_j_int, const.subfig_size*6)


def set_y_axis_label_for_leftmost_subplots(const, axs, pipeline):
    j = 0
    for i in range(len(const.used_variable_names)):
        i_name = const.used_variable_names[i]
        pipeline_i_int = const.name_to_list_position[i_name]
        axs[i][j].set_ylabel(
            "${}$".format(i_name), fontsize=const.subfig_size * 10)
        set_ytick_labels(
            axs[i][j], pipeline.lower_bound, pipeline.upper_bound,
            pipeline_i_int, const.subfig_size*6)


def make_color_bar(axs, const):
    cax, kw = mpl.colorbar.make_axes([ax for ax in axs])
    im = axs[1][0].get_images()[0]
    cbar = plt.colorbar(im, cax=cax, **kw, extend="both")
    cbar.ax.tick_params(labelsize=const.subfig_size * len(axs) * 2)


def get_output_file_name(method_name, pre_stamp, const, k):
    if "given" in method_name:
        return "results/{}_giv_{}_r{}_map.png".format(
            pre_stamp, const.data_stamp, const.resolution)
    elif "generated" in method_name:
        return "results/{}_gen_{}_{}_outN{}_r{}_map.png".format(
            pre_stamp, const.data_stamp, const.model_stamp,
            k, const.resolution)


def calc_map_given(x_pos, y_pos, resolution, grid_snapshots, labels, weights):
    x_ints = get_list_of_entries_at_pos(grid_snapshots, x_pos)
    y_ints = get_list_of_entries_at_pos(grid_snapshots, y_pos)
    weighted_label_map, weight_map = \
        make_weighted_label_and_weight_maps(
            resolution, x_ints, y_ints, labels, weights)
    return calculate_label_map(weighted_label_map, weight_map)


def get_list_of_entries_at_pos(grid_snapshots, pos):
    return grid_snapshots[:, pos]


def make_weighted_label_and_weight_maps(
        resolution, x_ints, y_ints, labels, weights):
    weighted_label_map = make_empty_map(resolution)
    weight_map = make_empty_map(resolution)
    for x_int, y_int, label, weight in zip(x_ints, y_ints, labels, weights):
        weighted_label_map[x_int][y_int] = \
            weighted_label_map[x_int][y_int] + label * weight
        weight_map[x_int][y_int] = weight_map[x_int][y_int] + weight
    return weighted_label_map, weight_map


def make_empty_map(resolution):
    return [[0 for y in range(resolution)] for x in range(resolution)]


def calculate_label_map(weighted_label_map, weight_map):
    return np.array([[[weighted_label_entry / weight_entry
                     if weight_entry > 0 else float("NaN")
                     for weighted_label_entry, weight_entry
                     in zip(weighted_label_row, weigth_row)]
                    for weighted_label_row, weigth_row
                    in zip(weighted_label_map, weight_map)]])


def calc_represented_map_generated(
        x_pos, y_pos, resolution, minima, maxima, model, representations):
    assert x_pos != y_pos, "x_pos and y_pos need to differ"
    print(x_pos, y_pos)
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


def calc_map_given_configurational_density(
        x_pos, y_pos, resolution, grid_snapshots, weights, fill_val=0):
    weight_map = [[0 for y in range(resolution)]
                  for x in range(resolution)]
    for nr, snapshot in enumerate(grid_snapshots):
        x_int = int(snapshot[x_pos])
        y_int = int(snapshot[y_pos])
        if x_int >= 0 and x_int <= resolution-1 and y_int >= 0 \
                and y_int <= resolution-1:
            weight_map[x_int][y_int] = weight_map[x_int][y_int] \
                + weights[nr]
    max_weight = np.amax(weight_map)
    weight_map = [[weight_map[i][j] / max_weight
                  if weight_map[i][j] > 0 else float("NaN")
                  for j in range(len(weight_map[i]))]
                  for i in range(len(weight_map))]
    return np.array([weight_map])


def plot_super_scatter(
        pipeline, const, pre_stamp, method,
        minima, maxima, max_row_len=6, **kwargs):
    """Generates a superfigure of scater plots.
    Iterates over the different dimensions and based on
    different input values for one dimensions
    as well as a fixed value fr all other dimensions,
    predicts the reconstructed value for that dimension.
    An optimal encoding and decoding will yield a diagonal
    line for each dimension indifferent of the value
    chosen for the other dimensions.
    """
    row_cnt = ((len(const.used_variable_names)-1)//max_row_len)+1
    fig, axs = plt.subplots(
        row_cnt, max_row_len,
        figsize=(
            const.subfig_size*max_row_len, const.subfig_size*row_cnt*1.3))
    fig.align_labels()
    for i, var_name in enumerate(const.used_variable_names):
        xs, ys = method(
            x_pos=const.used_name_to_list_position[var_name],
            resolution=const.resolution, minima=minima, maxima=maxima,
            **kwargs)
        if row_cnt > 1:
            new_axs = axs[i//max_row_len]
        else:
            new_axs = axs
        new_axs[i % max_row_len]\
            .scatter(xs, ys, s=const.subfig_size*20)
        new_axs[i % max_row_len]\
            .set_xlim(
                [minima[const.used_name_to_list_position[var_name]],
                 maxima[const.used_name_to_list_position[var_name]]])
        new_axs[i % max_row_len]\
            .set_ylim(
                [minima[const.used_name_to_list_position[var_name]],
                 maxima[const.used_name_to_list_position[var_name]]])
        new_axs[i % max_row_len]\
            .set_title(
                "${}$".format(var_name),
                fontsize=const.subfig_size*10)
        if i // max_row_len == (len(const.used_variable_names)-1) \
                // max_row_len:
            new_axs[i % max_row_len]\
                .set_xlabel(
                    "$Input$",
                    fontsize=const.subfig_size*5)
        if i % max_row_len == 0:
            new_axs[i % max_row_len]\
                .set_ylabel(
                    "$Reconstruction$",
                    fontsize=const.subfig_size*5)
        new_axs[i % max_row_len].tick_params(
            axis='both', which='both', top=False, labelbottom=True,
            bottom=True, left=True, labelleft=True)
        new_axs[i % max_row_len].set_xticks(
            np.linspace(min(xs), max(xs), 3))
        new_axs[i % max_row_len].set_yticks(
            np.linspace(min(xs), max(xs), 3))
        pipeline_var_int = const.name_to_list_position[var_name]
        axis_tick_labels = np.around(
            np.linspace(
                pipeline.lower_bound[pipeline_var_int],
                pipeline.upper_bound[pipeline_var_int],
                3),
            2)
        new_axs[i % max_row_len].set_xticklabels(
            axis_tick_labels, rotation=60, fontsize=const.subfig_size*4)
        new_axs[i % max_row_len].set_yticklabels(
            axis_tick_labels, fontsize=const.subfig_size*4)
    # if not all rows are filled
    # remove the remaining empty subplots in the last row
    if len(const.used_variable_names) % max_row_len != 0:
        for i in range(len(const.used_variable_names)
                       % max_row_len, max_row_len):
            new_axs[i].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig("results/{}_{}_{}_r{}_scat.png"
                .format(
                    pre_stamp, const.data_stamp, const.model_stamp,
                    const.resolution))
    plt.show()
    return


def calc_scatter_generated(
        x_pos, resolution, model, minima, maxima, fill_val=0):
    in_size = model.layers[0].output_shape[0][1]
    xs = np.linspace(minima[x_pos], maxima[x_pos], resolution)
    ys = []
    for x in xs:
        prediction = model.predict([[x if x_pos == pos_nr else fill_val
                                    for pos_nr in range(in_size)]])[0]
        ys.append(prediction[x_pos])
    return xs, ys


def calc_represented_scatter_generated(
        x_pos, resolution, model, minima, maxima, representations):
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
        pipeline, const, grid_snapshots, labels, weights, pre_stamp):
    plot_super_map(
        pipeline=pipeline, const=const, pre_stamp=pre_stamp,
        method=calc_map_given, grid_snapshots=grid_snapshots,
        labels=labels, weights=weights)


def plot_projected_example_paths(
        get_paths_function, const, pipeline, steps, model, pre_stamp):
    paths, labels = get_paths_function(const=const)
    projected_paths = [make_projected_path_from_path(
        pipeline=pipeline, path=path, const=const, steps=steps, model=model)
                       for path in paths]
    flattened_projected_paths = flatten_list_of_lists(projected_paths)
    projected_minimum = np.amin(
        np.transpose(flattened_projected_paths)[1], axis=0)
    projected_maximum = np.amax(
        np.transpose(flattened_projected_paths)[1], axis=0)
    model_output_name = model.output_names[0]
    ylim_bot = np.floor(projected_minimum)-0.1
    ylim_top = np.ceil(projected_maximum)+0.1
    for i in range(len(labels)):
        plot_projected_paths(
            projected_paths=projected_paths, labels=labels[:i+1],
            model_output_name=model_output_name, steps=steps,
            pre_stamp=pre_stamp, const=const, ylims=(ylim_bot, ylim_top))
    return projected_minimum, projected_maximum


def make_projected_path_from_path(pipeline, path, const, steps, model):
    out_size = model.layers[-1].output_shape[1]
    if out_size > 1:
        raise ValueError(
            "Data of dimensionality {} cannot be plotted".format(out_size))
    bn_path = pipeline.bound_normalize(path)
    bnr_path = pipeline.reduce(bn_path)
    path_len = len(bnr_path)
    projected_path = [model.predict([[bnr_path[int(path_len*i/(steps+1))]]])[0]
                      for i in range(steps+1)]
    return [[i, path[0]] for i, path in enumerate(projected_path)]


def plot_projected_paths(
        projected_paths, labels, model_output_name,
        steps, pre_stamp, const, ylims=(None, None)):
    for plot_path, label, i in zip(
            projected_paths, labels, range(len(labels))):
        plt.plot(
            *np.transpose(plot_path),
            label=str(label),
            color=const.plt_colors[i % len(labels)])
    plt.ylabel(model_output_name + " output")
    plt.xlabel(r"Progress along path [%]")
    plt.xticks(
        [steps*i/10 for i in range(11)],
        [100*i/10 for i in range(11)])
    plt.xlim(0, steps)
    plt.ylim(ylims)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.subplots_adjust(right=0.82)
    plt.savefig("results/{}_LatentSpacePath_plot_{}_{}D_{}.png".format(
        pre_stamp, const.model_stamp, const.bottleneck_size, len(labels)))
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
        x_int, y_int, const, pipeline, stamp, method,
        PES_function=lambda x: None, line_function=lambda w, x, y, z: None,
        line_formula=lambda x: np.float("NaN"), **kwargs):
    fig, ax = plt.subplots(1, 1)
    x_name = const.used_variable_names[x_int]
    pipeline_x_int = const.name_to_list_position[x_name]
    y_name = const.used_variable_names[y_int]
    pipeline_y_int = const.name_to_list_position[y_name]
    PES_function(const)
    line_function(line_formula, pipeline, pipeline_x_int, pipeline_y_int)
    method_name = function_to_str(method)
    if "density" not in method_name:
        cmap = const.label_cmap
    else:
        cmap = const.density_cmap
    plt.imshow(
        np.maximum(
            np.transpose(
                method(
                    x_pos=x_int, y_pos=y_int, resolution=const.resolution,
                    **kwargs)[0])[::-1],
            const.logvmin / 2),
        cmap=cmap,
        interpolation='nearest',
        norm=mpl.colors.LogNorm(
            vmin=const.logvmin,
            vmax=1.0-const.logvmin),
        extent=[0, 1, 0, 1],
        zorder=1)
    ax = set_xtick_labels(
        ax, pipeline.lower_bound, pipeline.upper_bound, pipeline_x_int,
        fontsize=const.subfig_size * 6)
    ax.set_xlabel(
        "${}$".format(x_name),
        fontsize=const.subfig_size * 10)
    ax = set_ytick_labels(
        ax, pipeline.lower_bound, pipeline.upper_bound, pipeline_y_int,
        fontsize=const.subfig_size * 6)
    ax.set_ylabel(
        "${}$".format(y_name),
        fontsize=const.subfig_size * 10)
    plt.colorbar(extend="both")
    plt.tight_layout()
    plt.savefig("results/{}_x{}_y_{}.png".format(stamp, x_int, y_int))
    plt.show()


def inject_PES(const):
    PES = plt.imread("PES_{}.png".format(const.dataSetType))
    PES[PES < 0.1] = np.nan
    plt.imshow(
        PES,
        extent=(0, 1, 0, 1),
        cmap=make_png_with_bad_as_transparent_colormap(),
        zorder=2)


def inject_dividing_line(function, pipeline, x_int, y_int):
    xs = np.linspace(
        pipeline.lower_bound[x_int],
        pipeline.upper_bound[x_int],
        11)
    ys = np.array([function(x) for x in xs])
    xs = (xs - pipeline.lower_bound[x_int]) \
        / (pipeline.upper_bound[x_int] - pipeline.lower_bound[x_int])
    ys = (ys - pipeline.lower_bound[y_int]) \
        / (pipeline.upper_bound[y_int] - pipeline.lower_bound[y_int])
    plt.plot(
        np.array(xs), np.array(ys), c="r")


def plot_reconstruction_from_latent_space(
        const, latent_minimum, latent_maximum,
        steps, recon_decoder, pre_stamp):
    fig = go.Figure()
    var_names = ["$"+name+"$" for name
                 in const.used_variable_names
                 + [const.used_variable_names[0]]]
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


def set_xtick_labels(ax, minima, maxima, x_int, fontsize):
    ax.tick_params(labelbottom=True, bottom=True,)
    ax.set_xticks(np.linspace(ax.dataLim.x0, ax.dataLim.x1, 3))
    ax.set_xticklabels(
        np.around(np.linspace(minima[x_int], maxima[x_int], 3), 2),
        rotation=60,
        fontsize=fontsize)
    return ax


def set_ytick_labels(ax, minima, maxima, y_int, fontsize):
    ax.tick_params(labelleft=True, left=True)
    ax.set_yticks(np.linspace(ax.dataLim.y0, ax.dataLim.y1, 3))
    ax.set_yticklabels(
        np.around(np.linspace(minima[y_int], maxima[y_int], 3), 2),
        fontsize=fontsize)
    return ax


def scatter_toy_path_with_potential(const, pipeline, path, label):
    bn_path = pipeline.bound_normalize(path)
    bn_min = (pipeline.lower_bound - pipeline.mean) / pipeline.std
    bn_max = (pipeline.upper_bound - pipeline.mean) / pipeline.std
    bn_inv_span = 1 / (bn_max - bn_min)
    bn_path = (bn_path - bn_min) * bn_inv_span
    bn_columns = np.transpose(bn_path)
    plt.scatter(bn_columns[0], bn_columns[1], s=2, zorder=3)
    inject_PES(const)
    plt.savefig("results/{}_{}_ScatterToyPath".format(
        const.dataSetType, label))


def plot_input_distribution(
        const, grid_snapshots, max_row_len, pipeline):
    cols = np.transpose(grid_snapshots)
    dimensions = len(grid_snapshots[0])
    row_cnt = ((dimensions-1)//max_row_len)+1
    fig, axs = plt.subplots(
        row_cnt, max_row_len,
        figsize=(
            const.subfig_size*max_row_len,
            const.subfig_size*row_cnt*1.3))
    for i in range(dimensions):
        if row_cnt > 1:
            new_axs = axs[i//max_row_len]
        else:
            new_axs = axs
        new_axs[i % max_row_len].hist(cols[i], const.resolution)
        new_axs[i % max_row_len].set_xlim(0, const.resolution-1)
        new_axs[i % max_row_len].set_xlabel(
                "${}$".format(const.used_variable_names[i]),
                fontsize=const.subfig_size * 10)
        i_name = const.used_variable_names[i]
        pipeline_i_int = const.name_to_list_position[i_name]
        new_axs[i % max_row_len].tick_params(
            axis="y", labelsize=const.subfig_size * 6)
        set_xtick_labels(
                new_axs[i % max_row_len], pipeline.lower_bound,
                pipeline.upper_bound, pipeline_i_int, const.subfig_size*6)
    # if not all rows are filled
    # remove the remaining empty subplots in the last row
    if dimensions % max_row_len != 0:
        for i in range(dimensions % max_row_len, max_row_len):
            new_axs[i].axis("off")
    fig.align_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(f"results/input_distribution_{const.data_stamp}.png")
