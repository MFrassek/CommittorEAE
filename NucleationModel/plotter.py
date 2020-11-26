import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
from helperFunctions import *
from autoEncoder import AutoEncoder


class DimensionalPosition():
    def __init__(self, const, i_int, j_int=None):
        self._resolution = const.resolution
        self._x_var_name = const.used_variable_names[i_int]
        self._x_dim = const.used_name_to_list_position[self._x_var_name]
        if isinstance(j_int, int):
            self._y_var_name = const.used_variable_names[j_int]
            self._y_dim = const.used_name_to_list_position[self._y_var_name]

    @property
    def resolution(self):
        return self._resolution

    @property
    def x_dim(self):
        return self._x_dim

    @property
    def y_dim(self):
        return self._y_dim


def make_super_map_plot(method, pipeline, pre_stamp, **kwargs):
    method_name = function_to_str(method)
    super_map = calculate_super_map(method, pipeline, **kwargs)
    plot_super_map(method_name, pipeline, super_map, pre_stamp)


def calculate_super_map(method, pipeline, **kwargs):
    return [[method(
            DimensionalPosition(pipeline.const, i, j),
            **kwargs) if j < i else []
          for j, _ in enumerate(pipeline.const.used_variable_names)]
          for i, _ in enumerate(pipeline.const.used_variable_names)]


def plot_super_map(method_name, pipeline, super_map, pre_stamp):
    cmap = select_color_map(method_name, pipeline.const)
    fig, axs = prepare_subplots(pipeline.const)
    make_subplot_heatmaps(super_map, axs, pipeline.const, cmap)
    make_axes_and_labels(pipeline, axs, fig)
    make_color_bar(axs, pipeline.const)
    plt.savefig(get_output_file_name(method_name, pre_stamp, pipeline.const))
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


def make_subplot_heatmaps(super_map, axs, const, cmap):
    for i, sub_map_row in enumerate(super_map):
        for j, sub_map in enumerate(sub_map_row):
            if j < i:
                axs[i][j].imshow(
                        np.maximum(sub_map[::-1], const.logvmin / 2),
                        cmap=cmap,
                        interpolation='nearest',
                        norm=mpl.colors.LogNorm(
                            vmin=const.logvmin, vmax=1 - const.logvmin),
                        extent=[0, 1, 0, 1])


def make_axes_and_labels(pipeline, axs, fig):
    remove_all_axis_labels(axs)
    remove_empty_subplot_axes(axs)
    set_x_axis_label_for_lowest_subplots(pipeline, axs)
    set_y_axis_label_for_leftmost_subplots(pipeline, axs)
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


def set_x_axis_label_for_lowest_subplots(pipeline, axs):
    i = len(pipeline.const.used_variable_names) - 1
    for j in range(i):
        print(i, j)
        axs[i][j].set_xlabel(
            f"${pipeline.const.used_variable_names[j]}$",
            fontsize=pipeline.const.subfig_size * 10)
        set_xtick_labels(axs[i][j], pipeline, j)


def set_y_axis_label_for_leftmost_subplots(pipeline, axs):
    j = 0
    for i in range(len(pipeline.const.used_variable_names)):
        axs[i][j].set_ylabel(
            f"${pipeline.const.used_variable_names[i]}$",
            fontsize=pipeline.const.subfig_size * 10)
        set_ytick_labels(axs[i][j], pipeline, i)


def make_color_bar(axs, const):
    cax, kw = mpl.colorbar.make_axes([ax for ax in axs])
    im = axs[1][0].get_images()[0]
    cbar = plt.colorbar(im, cax=cax, **kw, extend="both")
    cbar.ax.tick_params(labelsize=const.subfig_size * len(axs) * 2)


def get_output_file_name(method_name, pre_stamp, const):
    if "given" in method_name:
        return "results/{}_giv_{}_r{}_map.png".format(
            pre_stamp, const.data_stamp, const.resolution)
    elif "generated" in method_name:
        return "results/{}_gen_{}_{}_r{}_map.png".format(
            pre_stamp, const.data_stamp, const.model_stamp, const.resolution)


def calc_map_given(dim_position, grid_snapshots, labels, weights):
    print(dim_position.x_dim, dim_position.y_dim)
    x_ints = get_list_of_entries_at_pos(grid_snapshots, dim_position.x_dim)
    y_ints = get_list_of_entries_at_pos(grid_snapshots, dim_position.y_dim)
    weighted_label_map, weight_map = \
        make_weighted_label_and_weight_maps(
            dim_position.resolution, x_ints, y_ints, labels, weights)
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
    return np.array([[weighted_label_entry / weight_entry
                     if weight_entry > 0 else float("NaN")
                     for weighted_label_entry, weight_entry
                     in zip(weighted_label_row, weigth_row)]
                    for weighted_label_row, weigth_row
                    in zip(weighted_label_map, weight_map)])


def calc_represented_map_generated(
        dim_position, minmax_container, model, representations):
    def rescale_grid_point_means(grid_point_means):
        span_inv_resolution = \
            (minmax_container.r_maxima - minmax_container.r_minima) \
            / (dim_position.resolution - 1)
        return (grid_point_means * span_inv_resolution) \
            + minmax_container.r_minima

    print(dim_position.x_dim, dim_position.y_dim)
    xy_dimension_means = representations[
        (dim_position.x_dim, dim_position.y_dim)]
    return np.array([[model.predict(
        [[rescale_grid_point_means(xy_dimension_means[(j, i)])]])[0][0]
         if (j, i) in xy_dimension_means else float("NaN")
         for i in range(dim_position.resolution)]
        for j in range(dim_position.resolution)])


def calc_map_given_configurational_density(
        dim_position, grid_snapshots, weights):
    weight_map = make_empty_map(dim_position.resolution)
    x_ints = get_list_of_entries_at_pos(grid_snapshots, dim_position.x_dim)
    y_ints = get_list_of_entries_at_pos(grid_snapshots, dim_position.y_dim)
    for x_int, y_int, weight in zip(x_ints, y_ints, weights):
        weight_map[x_int][y_int] = weight_map[x_int][y_int] + weight
    max_weight = np.amax(weight_map)
    return np.array([[weight / max_weight
                      if weight > 0 else float("NaN")
                      for weight in weight_row]
                     for weight_row in weight_map])


def make_super_scatter_plot(
        method, pipeline, pre_stamp, max_row_len=6, **kwargs):
    super_scatter = calculate_super_scatter(method, pipeline, **kwargs)
    plot_super_scatter(pipeline, max_row_len, super_scatter, pre_stamp)


def calculate_super_scatter(method, pipeline, **kwargs):
    return [method(
        DimensionalPosition(pipeline.const, i), **kwargs)
        for i, _ in enumerate(pipeline.const.used_variable_names)]


def plot_super_scatter(pipeline, max_row_len, super_scatter, pre_stamp):
    fig, axs = prepare_subscatters(pipeline.const, max_row_len)
    fig.align_labels()
    make_subplot_scatters(super_scatter, axs, pipeline.const, max_row_len)
    set_labels_and_title_for_subscatters(pipeline, axs, max_row_len)
    set_x_axis_label_for_lowest_subscatters(pipeline.const, axs, max_row_len)
    set_y_axis_label_for_leftmost_subscatters(pipeline.const, axs, max_row_len)
    remove_empty_scatter_axes(pipeline.const, axs, max_row_len)
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(f"results/{pre_stamp}_{pipeline.const.data_stamp}_"
                + f"{pipeline.const.model_stamp}_"
                + f"r{pipeline.const.resolution}_scat.png")
    plt.show()


def prepare_subscatters(const, max_row_len):
    row_cnt = ((len(const.used_variable_names)-1)//max_row_len)+1
    fig, axs = plt.subplots(
        row_cnt, max_row_len,
        figsize=(const.subfig_size*max_row_len, const.subfig_size*row_cnt*1.3))
    if not isinstance(axs[0], np.ndarray):
        axs = [axs]
    return fig, axs


def make_subplot_scatters(super_scatter, axs, const, max_row_len):
    for i in range(len(const.used_variable_names)):
        axs[i//max_row_len][i % max_row_len].scatter(
            super_scatter[i][0], super_scatter[i][1], s=const.subfig_size * 20)


def set_labels_and_title_for_subscatters(pipeline, axs, max_row_len):
    for i in range(len(pipeline.const.used_variable_names)):
        print(i)
        axs[i // max_row_len][i % max_row_len].set_title(
            f"${pipeline.const.used_variable_names[i]}$",
            fontsize=pipeline.const.subfig_size * 10)
        set_xtick_labels(axs[i // max_row_len][i % max_row_len], pipeline, i)
        set_ytick_labels(axs[i // max_row_len][i % max_row_len], pipeline, i)


def set_x_axis_label_for_lowest_subscatters(const, axs, max_row_len):
    for i in range(max_row_len):
        axs[-1][i].set_xlabel("$Input$", fontsize=const.subfig_size * 5)


def set_y_axis_label_for_leftmost_subscatters(const, axs, max_row_len):
    for i in range((len(const.used_variable_names) + 1) // max_row_len):
        axs[i][0].set_ylabel("$Reconstruction$", fontsize=const.subfig_size * 5)


def remove_empty_scatter_axes(const, axs, max_row_len):
    if len(const.used_variable_names) % max_row_len != 0:
        for i in range(
                len(const.used_variable_names) % max_row_len, max_row_len):
            axs[-1][i].axis("off")


def calc_represented_scatter_generated(
        dim_position, model, minmax_container, representations):
    def rescale_grid_point_means(grid_point_means):
        span_inv_resolution = \
            (minmax_container.r_maxima - minmax_container.r_minima) \
            / (dim_position.resolution - 1)
        return (grid_point_means * span_inv_resolution) \
            + minmax_container.r_minima

    print(dim_position.x_dim)
    xs = np.linspace(
        minmax_container.r_minima[dim_position.x_dim],
        minmax_container.r_maxima[dim_position.x_dim],
        dim_position.resolution)
    x_dimension_means = representations[dim_position.x_dim]
    ys = np.array([model.predict(
        [[rescale_grid_point_means(x_dimension_means[(i,)])]])[0][0]
         if (i,) in x_dimension_means else float("NaN")
         for i in range(dim_position.resolution)])
    return xs, ys


def plot_loss_history(history, pre_stamp):
    plt.figure(figsize=(8, 8))
    for key, log_loss in history.history.items():
        plt.plot(range(1, 1+len(log_loss)), log_loss, label=key)
        plt.scatter(range(1, 1+len(log_loss)), log_loss, s=10)
    plt.ylim(0,)
    plt.xlim(1, len(log_loss))
    plt.xticks(range(1, 1+len(log_loss)))
    plt.legend(loc="lower right")
    plt.savefig(f"results/{pre_stamp}_loss_history.png")
    plt.show()


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
                    DimensionalPosition(const, x_int, y_int),
                    **kwargs))[::-1],
            const.logvmin / 2),
        cmap=cmap,
        interpolation='nearest',
        norm=mpl.colors.LogNorm(
            vmin=const.logvmin,
            vmax=1.0-const.logvmin),
        extent=[0, 1, 0, 1],
        zorder=1)
    set_xtick_labels(ax, pipeline, x_int)
    ax.set_xlabel(
        "${}$".format(x_name),
        fontsize=const.subfig_size * 10)
    set_ytick_labels(ax, pipeline, y_int)
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


def set_xtick_labels(ax, pipeline, index):
    ax.tick_params(labelbottom=True, bottom=True,)
    ax.set_xticks(np.linspace(ax.dataLim.x0, ax.dataLim.x1, 3))
    ax.set_xticklabels(
        np.around(np.linspace(
            pipeline.r_lower_bound[index], pipeline.r_upper_bound[index], 3),
            2),
        rotation=60,
        fontsize=pipeline.const.subfig_size * 6)


def set_ytick_labels(ax, pipeline, index):
    ax.tick_params(labelleft=True, left=True)
    ax.set_yticks(np.linspace(ax.dataLim.y0, ax.dataLim.y1, 3))
    ax.set_yticklabels(
        np.around(np.linspace(
            pipeline.r_lower_bound[index], pipeline.r_upper_bound[index], 3),
            2),
        fontsize=pipeline.const.subfig_size * 6)


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
            const.subfig_size*row_cnt))
    for i in range(dimensions):
        if row_cnt > 1:
            new_axs = axs[i//max_row_len]
        else:
            new_axs = axs
        new_axs[i % max_row_len].hist(cols[i], const.resolution)
        new_axs[i % max_row_len].set_xlim(0, const.resolution-1)
        new_axs[i % max_row_len].set_xlabel(
                "${}$".format(const.used_variable_names[i]),
                fontsize=const.subfig_size * 8)
        new_axs[i % max_row_len].tick_params(
            axis="y", labelsize=const.subfig_size * 4)
        set_xtick_labels(new_axs[i % max_row_len], pipeline, i)
    # if not all rows are filled
    # remove the remaining empty subplots in the last row
    if dimensions % max_row_len != 0:
        for i in range(dimensions % max_row_len, max_row_len):
            new_axs[i].axis("off")
    fig.align_labels()
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"results/input_distribution_{const.data_stamp}.png")


def plot_histogram_with_broken_axes(
        xs, bins, y_lower_1, y_upper_1, y_lower_2, y_upper_2, filename):
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    ax.hist(xs, bins)
    ax2.hist(xs, bins)
    ax.set_ylim(y_lower_2, y_upper_2)  # outliers only
    ax2.set_ylim(y_lower_1, y_upper_1)  # most of the data
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax2.set_xlabel("$p_B$", fontsize=12)
    ax2.set_ylabel("                              Count", fontsize=12)

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    plt.savefig(filename)
    plt.show()
