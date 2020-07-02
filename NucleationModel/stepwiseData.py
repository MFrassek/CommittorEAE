from autoEncoder import AutoEncoder
import tensorflow as tf
import numpy as np


class StepwiseData:
    def __init__(
            self, train_past_snapshots, train_snapshots,
            train_labels, train_weights,
            val_past_snapshots, val_snapshots,
            val_labels, val_weights):
        self._train_past_snapshots = train_past_snapshots
        self._train_snapshots = train_snapshots
        self._train_snapshot_cnt = len(train_past_snapshots)
        self._train_past_columns = np.transpose(train_past_snapshots)
        self._train_columns = np.transpose(train_snapshots)
        self._train_labels = train_labels
        self._train_weights = train_weights
        self._val_past_snapshots = val_past_snapshots
        self._val_snapshots = val_snapshots
        self._val_snapshot_cnt = len(val_past_snapshots)
        self._val_past_columns = np.transpose(val_past_snapshots)
        self._val_columns = np.transpose(val_snapshots)
        self._val_labels = val_labels
        self._val_weights = val_weights
        pass

    def get_score(
            self, used: list,
            name_to_list_position: dict,
            epochs: int,
            repetitions: int, const: object):
        """Generate datasets with only specific variables.
        Compile, train and evaluate a model repeatedly on these
        datasets and return the average loss obtained by the models.
        Params:
            used:
                list of parameters used
            epochs:
                number of epochs for a single training of a model
            repetitions:
                number of times a model will be trained and evaluated
            const:
                object of global constants
        """
        red_train_past_snapshots = []
        red_train_snapshots = []
        red_val_past_snapshots = []
        red_val_snapshots = []

        # add together the used columns
        for i in used:
            red_train_past_snapshots\
                .append(self._train_past_columns[name_to_list_position[i]])
            red_train_snapshots\
                .append(self._train_columns[name_to_list_position[i]])
            red_val_past_snapshots\
                .append(self._val_past_columns[name_to_list_position[i]])
            red_val_snapshots\
                .append(self._val_columns[name_to_list_position[i]])

        red_train_past_snapshots = np.transpose(red_train_past_snapshots)
        red_train_snapshots = np.transpose(red_train_snapshots)
        red_val_past_snapshots = np.transpose(red_val_past_snapshots)
        red_val_snapshots = np.transpose(red_val_snapshots)

        red_train_ds = tf.data.Dataset.from_tensor_slices(
            ({const.input_name: red_train_past_snapshots},
             {const.output_name_1: self._train_labels,
              const.output_name_2: red_train_snapshots},
             {const.output_name_1: self._train_weights,
              const.output_name_2: self._train_weights})) \
            .shuffle(self._train_snapshot_cnt) \
            .batch(const.batch_size)
        red_val_ds = tf.data.Dataset.from_tensor_slices(
            ({const.input_name: red_val_past_snapshots},
             {const.output_name_1: self._val_labels,
              const.output_name_2: red_val_snapshots},
             {const.output_name_1: self._val_weights,
              const.output_name_2: self._val_weights})) \
            .shuffle(self._val_snapshot_cnt) \
            .batch(const.batch_size)

        modelO = AutoEncoder(const)
        losses = []
        # repeatedly compile, train and validate a model
        # and record the loss
        for i in range(repetitions):
            print(" Repetition {}".format(i+1))
            model, _, _ = modelO.model(len(used))
            history = model.fit(
                red_train_ds,
                epochs=epochs,
                verbose=0,
                validation_data=red_val_ds)
            losses.append(history.history["val_label_loss"][-1])
        # return the mean loss of the repetitions
        return np.mean(losses)

    def bottom_up(
            self, used: list, unused: list,
            name_to_list_position: dict,
            param_limit: int,
            epochs: int, repetitions: int,
            const: object):
        """Recursively finds the most predictive variables from a set.
        In each call, all possible candidates from a list of unused
        variables are tested in combination with the allready selected
        variables, and the variable yielding the lowest average loss
        is added to the set of used variables before calling this
        function again. Once a set number of parameters is reached, the
        list of selected parameters as well as the corresponding loss
        are returned
        """
        losses = []
        # param limit may not exceed total number of variables available
        param_limit = min(param_limit, len(used)+len(unused))
        for i, unused_i in enumerate(unused):
            print(used+[unused_i])
            losses.append(self.get_score(
                used=used+[unused_i],
                name_to_list_position=name_to_list_position,
                epochs=epochs,
                repetitions=repetitions,
                const=const))
            print("  Mean loss: {}".format(losses[-1]))
        min_index = losses.index(min(losses))
        print("Added {}".format(unused[min_index]))
        if len(used) >= param_limit - 1 or len(unused) == 1:
            # Calculate score
            print("\n", used+[min_index], min(losses))
            return used+[unused[min_index]], min(losses)
        return self.bottom_up(
            used+[unused[min_index]],
            unused[:min_index]+unused[min_index+1:],
            name_to_list_position,
            param_limit,
            epochs,
            repetitions,
            const)

    def top_down(
            self, used: list, unused: list,
            name_to_list_position: dict,
            param_limit: int,
            epochs: int, repetitions: int,
            const: object):
        """Recursively finds the least predictive variables from a set.
        In each call, all possible sets where one variable is left out
        are thest, and the set yielding the lowest average loss
        is chosen (thereby removing the least informative variable)
        before calling this  function again. Once a set number of
        parameters is reached, the list of selected parameters as well
        as the corresponding loss are returned
        """
        losses = []
        # param limit may not exceed total number of variables available
        param_limit = min(param_limit, len(used)+len(unused))
        for i, _ in enumerate(used):
            print(used[:i]+used[i+1:])
            losses.append(self.get_score(
                used=used[:i]+used[i+1:],
                name_to_list_position=name_to_list_position,
                epochs=epochs,
                repetitions=repetitions,
                const=const))
            print("  Mean loss: {}".format(losses[-1]))
        min_index = losses.index(min(losses))
        print("Removed {}".format(used[min_index]))
        if len(used) <= param_limit + 1:
            # Calculate score
            print("\n", used[:min_index]+used[min_index+1:], min(losses))
            return used[:min_index]+used[min_index+1:], min(losses)
        return self.top_down(
            used[:min_index]+used[min_index+1:],
            unused+[used[min_index]],
            name_to_list_position,
            param_limit,
            epochs,
            repetitions,
            const)
