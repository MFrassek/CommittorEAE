from plotter import Plotter
from autoEncoder import AutoEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from helperFunctions import function_to_str
from data_read import get_one_TPS_path, get_one_TIS_path, get_one_toy_path
from os import listdir
import plotly.graph_objects as go


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
        reduced_list_var_names, reduced_name_to_list_position,
        pipeline, const, grid_snapshots, labels, weights, pre_stamp):
    Plotter.plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp=pre_stamp,
        method=Plotter.calc_map_given,
        grid_snapshots=grid_snapshots,
        labels=labels,
        weights=weights)


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
    Plotter.plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp=pre_stamp,
        method=Plotter.calc_map_generated,
        model=autoencoder_1,
        minima=minima,
        maxima=maxima)
    Plotter.plot_super_scatter(
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
    Plotter.plot_super_map(
        used_variable_names=reduced_list_var_names,
        name_to_list_position=reduced_name_to_list_position,
        lower_bound=pipeline.lower_bound,
        upper_bound=pipeline.upper_bound,
        const=const,
        pre_stamp="EncoderTest"+"_{}".format(function_to_str(loss_function)),
        method=Plotter.calc_map_generated,
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
                Plotter.calc_map_generated(
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
        pre_stamp,
        **kwargs):
    paths, labels = get_paths_function(const=const, **kwargs)
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


def flatten_list_of_lists(list_of_lists):
    return [y for x in list_of_lists for y in x]


def get_toy_paths(folder_name, const):
    paths = []
    labels = const.keep_labels
    for label in labels:
        paths.append(get_one_toy_path(folder_name, label))
    return paths, labels


def get_TPS_and_TIS_paths(const):
    paths = []
    labels = []
    TIS_labels = sorted(sorted(listdir(const.TIS_folder_name)), key=len)
    reformated_TIS_labels = \
        ["$MCG_{}$".format("{"+label[3:]+"}") for label in TIS_labels]
    labels.extend(reformated_TIS_labels)
    for label in TIS_labels:
        paths.append(get_one_TIS_path(const=const, interface=label))
    paths.append(get_one_TPS_path(const=const))
    labels.append("$TPS$")
    return paths, labels


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
        plt.ylabel(r"Fraction of path length [%]")
        plt.yticks(
            [steps*i/10 for i in range(11)],
            [100*i/10 for i in range(11)])
        plt.ylim(0, steps)
    else:
        plt.ylabel("$BN_2$")
    plt.title("Paths mapped onto the latent space")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.subplots_adjust(right=0.82)
    plt.savefig("results/{}_LatentSpacePath_plot_{}_{}D".format(
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
        stamp, method, **kwargs):
    fig, ax = plt.subplots(1, 1)
    plt.imshow(
        np.maximum(
            np.transpose(
                method(
                    x_pos=x_int,
                    y_pos=y_int,
                    resolution=const.resolution,
                    **kwargs)[0])[::-1],
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
