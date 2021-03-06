import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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

    @property
    def x_var_name(self):
        return self._x_var_name

    @property
    def y_var_name(self):
        return self._y_var_name


def make_super_map_plot(method, pipeline, **kwargs):
    super_map = calculate_super_map(method, pipeline, **kwargs)
    plot_super_map(method.__name__, pipeline, super_map)


def calculate_super_map(method, pipeline, **kwargs):
    return [[method(
            DimensionalPosition(pipeline.const, i, j),
            **kwargs) if j < i else []
          for j, _ in enumerate(pipeline.const.used_variable_names)]
          for i, _ in enumerate(pipeline.const.used_variable_names)]


def plot_super_map(method_name, pipeline, super_map):
    cmap = select_color_map(method_name, pipeline.const)
    fig, axs = prepare_subplots(pipeline.const)
    make_subplot_heatmaps(super_map, axs, pipeline.const, cmap)
    make_axes_and_labels(pipeline, axs, fig)
    make_color_bar(axs, pipeline.const)
    plt.savefig(get_output_basename(method_name, pipeline.const) + ".png")
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
                make_subplot_heatmap(axs[i][j], sub_map, const, cmap)


def make_subplot_heatmap(ax, heatmap, const, cmap):
    ax.imshow(
        np.maximum(heatmap[::-1], const.logvmin / 2),
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


def get_output_basename(method_name, const):
    if "given" in method_name:
        if "configurational" in method_name:
            return f"results/{const.dataSetType}_confgiv_{const.data_stamp}_" \
                + f"r{const.resolution}_map"
        else:
            return f"results/{const.dataSetType}_giv_{const.data_stamp}_" \
                + f"r{const.resolution}_map"
    elif "generated" in method_name:
        return f"results/{const.dataSetType}_gen_{const.data_stamp}_" \
            + f"{const.model_stamp}_r{const.resolution}_map"


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


def make_super_scatter_plot(method, pipeline, max_row_len=6, **kwargs):
    super_scatter = calculate_super_scatter(method, pipeline, **kwargs)
    plot_super_scatter(pipeline, max_row_len, super_scatter)


def calculate_super_scatter(method, pipeline, **kwargs):
    return [method(
        DimensionalPosition(pipeline.const, i), **kwargs)
        for i, _ in enumerate(pipeline.const.used_variable_names)]


def plot_super_scatter(pipeline, max_row_len, super_scatter):
    fig, axs = prepare_max_row_subplots(pipeline.const, max_row_len)
    fig.align_labels()
    make_subplot_scatters(super_scatter, axs, pipeline.const, max_row_len)
    set_labels_and_title_for_subscatters(pipeline, axs, max_row_len)
    set_x_axis_label_for_lowest_subscatters(pipeline.const, axs, max_row_len)
    set_y_axis_label_for_leftmost_subscatters(pipeline.const, axs, max_row_len)
    remove_empty_max_row_subplot_axes(pipeline.const, axs, max_row_len)
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(f"results/{pipeline.const.dataSetType}_"
                + f"{pipeline.const.data_stamp}_{pipeline.const.model_stamp}_"
                + f"r{pipeline.const.resolution}_scat.png")
    plt.show()


def prepare_max_row_subplots(const, max_row_len):
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


def remove_empty_max_row_subplot_axes(const, axs, max_row_len):
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


def make_loss_history_plot(history):
    plt.figure(figsize=(8, 8))
    for key, log_loss in history.history.items():
        plt.plot(range(1, 1+len(log_loss)), log_loss, label=key)
        plt.scatter(range(1, 1+len(log_loss)), log_loss, s=10)
    plt.ylim(0,)
    plt.xlim(1, len(log_loss))
    plt.xticks(range(1, 1+len(log_loss)))
    plt.legend(loc="lower right")
    plt.savefig(f"results/{history.model.name}_LossHistory_"
                + f"e{len(history.history['loss'])}.png")
    plt.show()


def make_projected_path_plot(pipeline, model):
    projected_paths, labels = \
        get_projected_paths_and_labels(pipeline, model)
    model_output_name = model.output_names[0]
    plot_projected_paths(
        projected_paths=projected_paths, labels=labels,
        model_output_name=model_output_name, const=pipeline.const)


def get_projected_paths_and_labels(pipeline, model):
    paths, labels = pipeline.const.path_getter_function(const=pipeline.const)
    projected_paths = [make_projected_path_from_path(model, pipeline, path)
                       for path in paths]
    return projected_paths, labels


def make_projected_path_from_path(model, pipeline, path):
    out_size = model.layers[-1].output_shape[1]
    if out_size > 1:
        raise ValueError(
            f"Data of dimensionality {out_size} cannot be plotted")
    bnr_path = pipeline.reduce(pipeline.bound_normalize(path))
    path_len = len(bnr_path)
    steps = pipeline.const.projection_steps
    projected_path = [model.predict([[bnr_path[int(path_len*i/(steps+1))]]])[0]
                      for i in range(steps+1)]
    return [[i, path[0]] for i, path in enumerate(projected_path)]


def plot_projected_paths(projected_paths, labels, model_output_name, const):
    label_cnt = len(labels)
    steps = const.projection_steps
    for plot_path, label, color in zip(projected_paths, labels, const.plt_colors):
        plt.plot(*np.transpose(plot_path), label=str(label), color=color)
    plt.ylabel(model_output_name + " output")
    plt.xlabel(r"Progress along path [%]")
    plt.xticks([steps*i/10 for i in range(11)], [100*i/10 for i in range(11)])
    plt.xlim(0, steps)
    projected_minimum, projected_maximum = \
        get_low_and_high_point_of_projected_paths(projected_paths)
    plt.ylim(np.floor(projected_minimum)-0.1, np.ceil(projected_maximum)+0.1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.subplots_adjust(right=0.82)
    plt.savefig(f"results/{const.dataSetType}_{model_output_name}Projection_"
                + f"{const.model_stamp}_{const.bottleneck_size}D_{label_cnt}.png")
    plt.show()


def get_projected_minimum_and_maximum(pipeline, model):
    projected_paths, _ = get_projected_paths_and_labels(pipeline, model)
    return get_low_and_high_point_of_projected_paths(projected_paths)


def get_low_and_high_point_of_projected_paths(paths):
    flattened_paths = flatten_list_of_lists(paths)
    low_point = np.amin(
        np.transpose(flattened_paths)[1], axis=0)
    high_point = np.amax(
        np.transpose(flattened_paths)[1], axis=0)
    return low_point, high_point


def flatten_list_of_lists(list_of_lists):
    return [y for x in list_of_lists for y in x]


def make_relative_importance_plot(encoder, const):
    relative_importances = get_relative_encoder_importances(encoder)
    plot_relative_importances(const.used_variable_names, relative_importances)


def get_relative_encoder_importances(encoder):
    return make_components_normalized(
        make_components_one_dimensional(
            make_components_absolute(
                get_encoder_formula_components(encoder))))


def get_encoder_formula_components(encoder):
    """Calculates the linear formula represented by the encoder."""
    in_size = encoder.layers[0].output_shape[0][1]
    base_predictions = encoder.predict([[np.zeros(in_size)]])[0]
    return np.transpose(np.array([encoder.predict([
                   [1 if i == dim else 0 for i in range(in_size)]])[0]
                   for dim in range(in_size)]) - base_predictions)


def make_components_absolute(formula_components):
    return np.array([list(map(abs, component))
                    for component in formula_components])


def make_components_one_dimensional(formula_components):
    return np.sum(formula_components, axis=0)


def make_components_normalized(formula_components):
    return formula_components / sum(formula_components)


def plot_relative_importances(variable_names, importances):
    dollar_names = [f"${variable_name}$" for variable_name in variable_names]
    importance_precentages = [importance * 100 for importance in importances]
    plt.bar(dollar_names, importance_precentages)
    plt.ylabel("Relative importance [%]")
    plt.ylim(0, 100)
    plt.xticks(rotation=60)
    plt.savefig("results/LinearComponents.png")
    plt.show()


def make_single_map_plot(
        dim_position, pipeline, method, line_formula=None, **kwargs):
    fig, ax = plt.subplots(1, 1)
    try:
        inject_dividing_line(line_formula, pipeline, dim_position)
    except TypeError:
        pass
    cmap = select_color_map(method.__name__, pipeline.const)
    heatmap = np.transpose(method(dim_position, **kwargs))
    make_subplot_heatmap(ax, heatmap, pipeline.const, cmap)
    make_single_map_labels_and_tick_labels(ax, pipeline, dim_position)
    make_color_bar([[ax], [ax]], pipeline.const)
    plt.savefig(
        get_output_basename(method.__name__, pipeline.const)
        + f"_x{dim_position.x_dim}_y_{dim_position.y_dim}.png")
    plt.show()


def inject_dividing_line(function, pipeline, dim_position):
    x_dim = dim_position.x_dim
    y_dim = dim_position.y_dim
    xs = np.linspace(
        pipeline.r_lower_bound[x_dim], pipeline.r_upper_bound[x_dim], 100)
    ys = np.array([function(x) for x in xs])
    xs = (xs - pipeline.r_lower_bound[x_dim]) \
        / (pipeline.r_upper_bound[x_dim] - pipeline.r_lower_bound[x_dim])
    ys = (ys - pipeline.r_lower_bound[y_dim]) \
        / (pipeline.r_upper_bound[y_dim] - pipeline.r_lower_bound[y_dim])
    where_y_within_range = np.where((ys >= 0) & (ys <= 1))
    plt.plot(xs[where_y_within_range], ys[where_y_within_range], c="r")


def calculate_slope_MCG_BigCage(x):
    return 68.14 - 0.4286*x


def calculate_slope_now_BigCage(x):
    return 30 - 0.0553*x


def make_single_map_labels_and_tick_labels(ax, pipeline, dim_position):
    set_xtick_labels(ax, pipeline, dim_position.x_dim)
    ax.set_xlabel(
        f"${dim_position.x_var_name}$",
        fontsize=pipeline.const.subfig_size * 10)
    set_ytick_labels(ax, pipeline, dim_position.y_dim)
    ax.set_ylabel(
        f"${dim_position.y_var_name}$",
        fontsize=pipeline.const.subfig_size * 10)


def make_representative_path_plot(const, latent_minmax, reconstruction_decoder):
    def add_trace(prediction, i):
        return go.Scatterpolar(
            r=np.append(prediction, prediction[0]),
            theta=var_names+[var_names[0]],
            name=f"{latent_linspace[i]:.1f}",
            showlegend=True,
            line=dict(color="rgb({},{},{})".format(
                0.8 - 0.6 * i / const.unprojection_steps,
                0.2 + 0.6 * i / const.unprojection_steps,
                0.2 + 0.6 * i / const.unprojection_steps)))

    latent_linspace = np.linspace(
        np.floor(latent_minmax[0]), np.ceil(latent_minmax[1]),
        const.unprojection_steps)
    var_names = [f"${name}$" for name in const.used_variable_names]
    predictions = [reconstruction_decoder.predict([latent_value])[0]
                   for latent_value in latent_linspace]
    fig = go.Figure()
    for i, prediction in enumerate(predictions):
        fig.add_trace(add_trace(prediction, i))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, tickangle=0),
            angularaxis=dict(tickfont=dict(size=18))),
        legend_title_text="$\ BN_1 input$")
    fig.write_image(f"results/{const.dataSetType}_PathReconstruction_"
                    + f"{const.model_stamp}.png")
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


def make_input_distribution_plot(grid_snapshots, max_row_len, pipeline):
    fig, axs = prepare_max_row_subplots(pipeline.const, max_row_len)
    make_subplot_hists(grid_snapshots, pipeline, axs, max_row_len)
    set_axis_labels_and_tick_labels_for_superhist(pipeline, axs, max_row_len)
    remove_empty_max_row_subplot_axes(pipeline.const, axs, max_row_len)
    fig.align_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(f"results/input_distribution_{pipeline.const.data_stamp}.png")


def make_subplot_hists(grid_snapshots, pipeline, axs, max_row_len):
    cols = np.transpose(grid_snapshots)
    for i in range(len(pipeline.const.used_variable_names)):
        axs[i//max_row_len][i % max_row_len].hist(
            cols[i], pipeline.const.resolution)


def set_axis_labels_and_tick_labels_for_superhist(pipeline, axs, max_row_len):
    for i in range(len(pipeline.const.used_variable_names)):
        axs[i//max_row_len][i % max_row_len].set_xlim(
            0, pipeline.const.resolution-1)
        axs[i//max_row_len][i % max_row_len].set_xlabel(
                f"${pipeline.const.used_variable_names[i]}$",
                fontsize=pipeline.const.subfig_size * 8)
        axs[i//max_row_len][i % max_row_len].tick_params(
            axis="y", labelsize=pipeline.const.subfig_size * 4)
        set_xtick_labels(axs[i//max_row_len][i % max_row_len], pipeline, i)


def make_histogram_with_broken_axis_plot(xs, bins, lower_range, upper_range):
    f, (upper_ax, lower_ax) = plt.subplots(2, 1, sharex=True)
    make_broken_hist_upper_half(upper_ax, xs, bins, upper_range)
    make_broken_hist_lower_half(lower_ax, xs, bins, lower_range)
    make_diagonal_break_lines(upper_ax, lower_ax)
    plt.savefig("results/pB_distribution.png")
    plt.show()


def make_broken_hist_upper_half(upper_ax, xs, bins, upper_range):
    upper_ax.hist(xs, bins)
    upper_ax.set_ylim(upper_range)
    upper_ax.spines['bottom'].set_visible(False)
    upper_ax.tick_params(labelbottom=False, bottom=False)


def make_broken_hist_lower_half(lower_ax, xs, bins, lower_range):
    lower_ax.hist(xs, bins)
    lower_ax.set_ylim(lower_range)
    lower_ax.spines['top'].set_visible(False)
    lower_ax.set_xlabel("$p_B$", fontsize=12)
    lower_ax.set_ylabel(30 * " " + "Count", fontsize=12)


def make_diagonal_break_lines(upper_ax, lower_ax):
    line_len = 0.015
    kwargs = dict(transform=upper_ax.transAxes, color='k', clip_on=False)
    upper_ax.plot((-line_len, +line_len), (-line_len, +line_len), **kwargs)
    upper_ax.plot((1 - line_len, 1 + line_len), (-line_len, +line_len), **kwargs)
    kwargs = dict(transform=lower_ax.transAxes, color='k', clip_on=False)
    lower_ax.plot((-line_len, +line_len), (1 - line_len, 1 + line_len), **kwargs)
    lower_ax.plot((1 - line_len, 1 + line_len), (1 - line_len, 1 + line_len), **kwargs)
