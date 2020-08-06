from plotter import Plotter
from autoEncoder import AutoEncoder
import tensorflow as tf
import matplotlib.pyplot as plt


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
        enocoder, decoder_1, decoder_2 = \
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
