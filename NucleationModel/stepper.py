from autoEncoder import AutoEncoder
from losses import *
import tensorflow as tf
import numpy as np


class Stepper:
    @staticmethod
    def iter_top_down(
            pipeline, train_dataset, val_dataset, param_limit, repetitions):
        """Iteratively finds the least informative input dimension and removes
        it until only a predefined number of input dimensions is left.
        At each iteration generates n datasets, all leaving out one of the
        remaining n variables, trains and tests a model on these datasets.
        Uses the variables used for the best performing model for the
        next iteration, and thereby removes the least informative variable at
        each step.
        Caution: This function has side effects, as it updates the pipeline
        and the const object stored in it."""
        train_bn_snapshots = pipeline.bound_normalize(train_dataset.snapshots)
        val_bn_snapshots = pipeline.bound_normalize(val_dataset.snapshots)
        removed_variables = []
        min_losses = []
        used = pipeline.const.used_variable_names.copy()
        for i in range(max(0, len(used) - param_limit)):
            losses = []
            for j, excludee in enumerate(used):
                print(f"Leaving out {excludee}")
                pipeline.const.used_variable_names = used[:j]+used[j+1:]
                train_ds, val_ds = pipeline.prepare_stepper(
                    train_bn_snapshots=train_bn_snapshots,
                    train_dataset=train_dataset,
                    val_bn_snapshots=val_bn_snapshots,
                    val_dataset=val_dataset)
                losses.append(Stepper.get_score(
                    train_ds, val_ds, repetitions, pipeline.const))
                print(f"  Mean label loss: {losses[-1]:.3f}")
            min_index = losses.index(min(losses))
            print(f"\nRemoved {used[min_index]}\t{len(used)-1} variables left\n")
            removed_variables.append(used[min_index])
            min_losses.append(min(losses))
            used = used[:min_index]+used[min_index+1:]
            print(f"{used}\n")
        print(f"\nFinal set: {used}\nFinal loss: {min(losses)}\nRemoved "
              + f"variables: {removed_variables}\nLosses at each step{min_losses}")
        return used, min(losses), removed_variables, min_losses

    @staticmethod
    def get_score(train_ds, val_ds, repetitions, const):
        losses = []
        for i in range(repetitions):
            autoencoder, _, _, _, _, _ = \
                AutoEncoder.make_models(const)
            history = autoencoder.fit(
                x=train_ds,
                epochs=const.epochs,
                verbose=0,
                validation_data=val_ds,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3)])
            losses.append(history.history["val_loss"][-1])
        return np.mean(losses)
