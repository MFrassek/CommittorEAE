import numpy as np
from copy import deepcopy
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt


class ImportanceData():
    def __init__(
            self, past_snapshots, snapshots, labels, weights,
            corr_thresholds):
        self._past_snapshots = past_snapshots
        self._snapshots = snapshots
        self._snapshot_cnt = len(past_snapshots)
        self._columns = np.transpose(past_snapshots)
        self._labels = labels
        self._weights = weights
        self._strong_corr_threshold = corr_thresholds[0]
        self._weak_corr_threshold = corr_thresholds[1]

    def measure_correlation(self):
        covar_matrix = np.cov(self._columns)
        strong_corr_inputs = []
        weak_corr_inputs = []
        for row_nr in range(len(covar_matrix)):
            for entry_nr in range(len(covar_matrix[row_nr])):
                if row_nr > entry_nr:
                    if abs(covar_matrix[row_nr][entry_nr]) \
                            >= self._strong_corr_threshold:
                        strong_corr_inputs.append(
                            [[str(row_nr), str(entry_nr)],
                             "{:.3f}".format(covar_matrix[row_nr][entry_nr])])
                    elif abs(covar_matrix[row_nr][entry_nr]) \
                            >= self._weak_corr_threshold:
                        weak_corr_inputs.append(
                            [[str(row_nr), str(entry_nr)],
                             "{:.3f}".format(covar_matrix[row_nr][entry_nr])])
        if len(strong_corr_inputs) > 0 or len(weak_corr_inputs) > 0:
            print(("Caution!\nCorrelation between input data can affect the "
                  + "reliability of the importance measure.\n"
                  + "Strong correlations of more than {} "
                  + "were found between {} pair(s) of input variables:\n\t{}\n"
                  + "Additionally, weak correlations of more than "
                  + "{} were found between {} pair(s) of input variables:\n\t")
                  .format(
                  self._strong_corr_threshold),
                  len(strong_corr_inputs),
                  self._weak_corr_threshold,
                  len(weak_corr_inputs),
                  "\n\t".join([": ".join([",".join(subentry)
                               if isinstance(subentry, list) else subentry
                               for subentry in entry])
                               for entry in strong_corr_inputs]),
                  "\n\t".join([": ".join([",".join(subentry)
                               if isinstance(subentry, list) else subentry
                               for subentry in entry])
                               for entry in weak_corr_inputs]))
        else:
            print("No correlation above {} found between the inputs."
                  .format(weak_corr_threshold))
        return strong_corr_inputs, weak_corr_inputs

    def perturb_snapshots(self, mod_along, perturbation):
        transposed = deepcopy(self._columns)
        rand_array = np.random.uniform(
            1 - perturbation,
            1 + perturbation,
            self._snapshot_cnt)
        transposed[mod_along] = self._columns[mod_along] * rand_array
        return np.transpose(transposed)

    def set_mean_snapshots(self, mod_along, column_mean):
        transposed = deepcopy(self._columns)
        transposed[mod_along] = column_mean
        return np.transpose(transposed)

    def HIPR_snapshots(self, mod_along, min_value, max_value):
        transposed = deepcopy(self._columns)
        rand_array = np.random.uniform(
            min_value,
            max_value,
            self._snapshot_cnt)
        transposed[mod_along] = rand_array
        return np.transpose(transposed)

    def shuffle_snapshots(self, mod_along):
        # since independent runs should use different shuffled lists,
        # nothing is passed down
        transposed = deepcopy(self._columns)
        transposed[mod_along] = shuffle(transposed[mod_along])
        return np.transpose(transposed)

    def calc_importance(
            self, mode: str, mode_var, model, val_ds,
            i_s: list, repetitions: int, batch_size: int):
        eval_steps = self._snapshot_cnt//batch_size
        orig_loss = model.evaluate(val_ds, verbose=0, steps=eval_steps)
        meta_losses = []
        # initialization dependent on the mode
        if mode == "Perturb":
            perturbation = mode_var
        elif mode == "Mean":
            if repetitions > 1:
                print("The mean mode does not entail stochasticity. \nNumber"
                      + " of repetitions was set to '1' for this measurement.")
                repetitions = 1
            mean_value_array = np.mean(self._past_snapshots, axis=0)
        elif mode == "HIPR":
            min_value = mode_var[0]
            max_value = mode_var[1]

        print("Mode: {}".format(mode))
        for repetition in range(repetitions):
            print("Repetition {}.".format(repetition+1))
            losses = [[] for i in range(len(orig_loss))]
            for variable_nr in i_s:
                print("\tPerturbing variable {}.".format(variable_nr))
                # generating modified snapshots
                if mode == "Perturb":
                    mod_snapshots = self.perturb_snapshots(
                        variable_nr,
                        perturbation)
                elif mode == "Mean":
                    mod_snapshots = self.set_mean_snapshots(
                        variable_nr,
                        mean_value_array[variable_nr])
                elif mode == "HIPR":
                    mod_snapshots = self.HIPR_snapshots(
                        variable_nr,
                        min_value, max_value)
                elif mode == "Shuffle":
                    mod_snapshots = self.shuffle_snapshots(variable_nr)

                mod_val_ds = tf.data.Dataset.from_tensor_slices(
                    ({model.input_names[0]: mod_snapshots},
                     {model.output_names[0]: self._labels,
                     model.output_names[1]: self._snapshots},
                     {model.output_names[0]: self._weights,
                     model.output_names[1]: self._weights}))\
                    .shuffle(self._snapshot_cnt).batch(batch_size)
                # calculate the different losses with the new dataset
                mod_loss = \
                    model.evaluate(mod_val_ds, verbose=0, steps=eval_steps)
                # append the losses to a collective list for later comparison
                for i in range(len(orig_loss)):
                    losses[i].append(max(0, mod_loss[i] - orig_loss[i]))
            # average over the loss lists
            # negative increases of loss are set to zero
            for row_nr in range(len(losses)):
                full_loss = sum(losses[row_nr])
                for col_nr in range(len(losses[row_nr])):
                    if full_loss > 0:
                        losses[row_nr][col_nr] = \
                            losses[row_nr][col_nr]/full_loss
                    else:
                        losses[row_nr][col_nr] = 0
            meta_losses.append(np.array(losses))

        tot_norm_losses = np.transpose([sum(np.transpose(sum(meta_losses)))])
        # sets value to 1 if all the losses add up to 0
        # which would cause a divide by zero error
        tot_norm_losses = np.array([value[0] if value != 0 else 1
                                   for value in tot_norm_losses])
        return sum(meta_losses)/np.transpose([tot_norm_losses])

    def plot_super_importance(
            self, subfig_size: int, i_s: list,
            stamp: str, var_names: dict, repetitions: int,
            modes: list, loss_names: list, val_ds, model):
        # takes the first dimension of the first batch of val_ds which gives
        # the batch size
        batch_size = np.shape(list(
            val_ds.as_numpy_iterator())[0][0][model.input_names[0]])[0]
        fig, axs = plt.subplots(
            len(loss_names),
            len(modes),
            figsize=(
                subfig_size*len(modes),
                subfig_size*len(loss_names)))
        fig.suptitle(
            "Input importance measures",
            fontsize=subfig_size*4, y=1.04 - 0.04*len(loss_names))
        for mode_nr in range(len(modes)):
            losses = self.calc_importance(
                *modes[mode_nr], model, val_ds,
                i_s, repetitions, batch_size)
            # Takes care of different handling of subplots if
            # there is one vs several rows/columns.
            for loss_nr in range(len(loss_names)):
                if len(modes) == 1:
                    if len(loss_names) == 1:
                        new_axs = axs
                    else:
                        new_axs = axs[loss_nr]
                else:
                    if len(loss_names) == 1:
                        new_axs = axs[mode_nr]
                    else:
                        new_axs = axs[loss_nr][mode_nr]
                new_axs.bar(range(len(i_s)), losses[loss_nr])
                if loss_nr == 0:
                    new_axs.set_title(
                        modes[mode_nr][0],
                        fontsize=subfig_size*3)
                if mode_nr == 0:
                    new_axs.set_ylabel(
                        "{}".format(loss_names[loss_nr]),
                        fontsize=subfig_size*3)
                new_axs.tick_params(
                    axis='x',
                    which='major',
                    labelsize=subfig_size/len(i_s)**0.5*9,
                    rotation=75)
                new_axs.tick_params(
                    axis='y',
                    which='major',
                    labelsize=subfig_size*3)
        plt.setp(
            axs, xticks=range(len(i_s)),
            xticklabels=["$"+var_names[i]+"$" for i in i_s],
            yticks=[0, 1], yticklabels=[0, 1])

        plt.savefig("{}_r_{}_importance.png".format(stamp, repetitions,))
        plt.show()
