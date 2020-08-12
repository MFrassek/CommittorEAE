from plotter import Plotter
from autoEncoder import AutoEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from helperFunctions import function_to_str


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
        AutoEncoder.model(len(reduced_list_var_names), loss_function, const)
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
        AutoEncoder.model(len(reduced_list_var_names), loss_function, const)
    history = autoencoder.fit(
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

def map_path_on_2D_latent_space(
        pipeline, path, encoder, skip, pre_stamp, const):
    processed_path = pipeline.rbn(path)
    predictions = []
    inv_path_len = 1/len(path)
    for i, snapshot in enumerate(processed_path[::skip]):
        predictions.append([*encoder.predict([[snapshot]])[0],
                      (i*inv_path_len*skip)])
    for prediction in predictions:
        plt.scatter(
            *prediction[:2],
            c=[
                [1-prediction[2],
                0,
                prediction[2]]],
            s=10)
    plt.xlim(-10, 10)
    plt.xlabel("$BN_1$")
    plt.ylim(-10, 10)
    plt.ylabel("$BN_2$")
    plt.title("Path mapping onto the latent space")
    plt.savefig("results/LatentSpacePath_scat_{}_{}".format(
        pre_stamp, const.model_stamp))
    plt.show()
    plt.plot(*np.transpose(predictions)[:2], c="b")
    plt.xlabel("$BN_1$")
    plt.ylabel("$BN_2$")
    plt.title("Path mapping onto the latent space")
    plt.savefig("results/2DLatentSpacePath_plot_{}_{}".format(
        pre_stamp, const.model_stamp))
    plt.show()

def map_path_on_1D_latent_space(
        pipeline, path, encoder, skip, pre_stamp, const):
    processed_path = pipeline.rbn(path)
    predictions = []
    inv_path_len = 1/len(path)
    for i, snapshot in enumerate(processed_path[::skip]):
        predictions.append([*encoder.predict([[snapshot]])[0],
                      (i*inv_path_len*skip)])
    for prediction in predictions:
        plt.scatter(
            prediction[0], 1,
            c=[
                [1-prediction[1],
                0,
                prediction[1]]],
            s=10)
    plt.xlim(-10, 10)
    plt.xlabel("$BN_1$")
    plt.yticks([])
    plt.title("Path mapping onto the latent space")
    plt.savefig("results/1DLatentSpacePath_scat_{}_{}".format(
        pre_stamp, const.model_stamp))
    plt.show()

def map_path_on_timed_1D_latent_space(
        pipeline, path, encoder, skip, pre_stamp, const):
    processed_path = pipeline.rbn(path)
    predictions = []
    for i, snapshot in enumerate(processed_path[::skip]):
        predictions.append([*encoder.predict([[snapshot]])[0],
                      i])
    plt.plot(*np.transpose(predictions), c = "b")
    plt.xlim(-10, 10)
    plt.xlabel("$BN_1$")
    plt.ylabel("Time [x{} snapshots]".format(skip))
    plt.title("Path mapping onto the latent space")
    plt.savefig("results/1DTimedLatentSpacePath_plot_{}_{}".format(
        pre_stamp, const.model_stamp))
    plt.show()