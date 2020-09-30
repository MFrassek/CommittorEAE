from reducer import Reducer
from bounder import Bounder
from normalizer import Normalizer
from gridifier import Gridifier
from pB_approximator import pB_Approximator
from trimmer import Trimmer
from trimmer import HalfTrimmer
from pB_balancer import pB_Balancer
from hypercube_balancer import Hypercube_Balancer
from multidim_balancer import MultiDim_Balancer
from squeezer import Squeezer
from corrector import Corrector


import numpy as np
import tensorflow as tf

class Pipeline():
    def __init__(self, const, base_snapshots):
        self._const = const
        self._snapshot_cnt = len(base_snapshots)
        self._bounder = Bounder(base_snapshots, self._const.outlier_cutoff)
        base_snapshots = self._bounder.bound_snapshots(base_snapshots)
        self._normalizer = Normalizer(base_snapshots)
        base_snapshots = self._normalizer.normalize_snapshots(base_snapshots)
        self._minima = np.amin(base_snapshots, axis=0)
        self._maxima = np.amax(base_snapshots, axis=0)

    @property
    def lower_bound(self):
        return self._bounder.lower_bound

    @property
    def upper_bound(self):
        return self._bounder.upper_bound

    @property
    def mean(self):
        return self._normalizer.mean

    @property
    def std(self):
        return self._normalizer.std

    @property
    def minima(self):
        return self._minima

    @property
    def maxima(self):
        return self._maxima

    @property
    def snapshot_cnt(self):
        return self._snapshot_cnt

    def bound_normalize(self, snapshots):
        """Bound and normalize snapshots."""
        snapshots = self._bounder.bound_snapshots(snapshots)
        snapshots = self._normalizer.normalize_snapshots(snapshots)
        return snapshots

    def reduce(self, snapshots, reduced_list_var_names):
        """Reduce snapshots."""
        reducer = Reducer(
            reduced_list_var_names,
            self._const.name_to_list_position)
        snapshots = reducer.reduce_snapshots(snapshots)
        return snapshots

    def gridify(self, snapshots):
        """Gridify snapshots."""
        gridifier = Gridifier(snapshots, self._const.resolution)
        snapshots = gridifier.gridify_snapshots(snapshots)
        return snapshots

    def approximate(self, snapshots, dataset):
        """Approximate pBs."""
        pB_dict, pBs, pB_weights = pB_Approximator.approximate_pBs(
            snapshots,
            dataset.labels,
            dataset.weights)
        return pB_dict, pBs, pB_weights

    def trim(self, pBs, *args):
        """Trim inputs based on pBs."""
        trimmer = Trimmer(pBs)
        trimmed = [trimmer.trim_snapshots(arg) for arg in args]
        return trimmed

    def squeeze(self, pBs):
        """Squeeze, i.e. replace pB = 1 with lower value."""
        pBs = Squeezer.squeeze_pBs(pBs, self._const)
        return pBs

    def normalize(self, snapshots):
        """Normalize snapshots."""
        normalizer = Normalizer(snapshots)
        snapshots = normalizer.normalize_snapshots(snapshots)
        return snapshots

    def pB_balance(self, pBs):
        pBb_weights = \
            pB_Balancer.balance(pBs, self._const.balance_bins)
        return pBb_weights

    def hypercube_balance(self, snapshots):
        hcb_weights = \
            Hypercube_Balancer.balance(
            snapshots, self._const.balance_bins)
        return hcb_weights

    def multidim_balance(self, snapshots):
        mdb_weights = \
            MultiDim_Balancer.balance(
                snapshots, self._const.balance_bins)
        return mdb_weights

    def correct_1D(self, snapshots):
        corrected_1D = Corrector.correct_1D_row(snapshots)
        return corrected_1D

    def correct_2D(self, snapshots):
        corrected_2D = Corrector.correct_2D_grid(snapshots)
        return corrected_2D

    def pack_tf_dataset(
            self,
            snapshots,
            labels,
            prediction_weights,
            reconstruction_weights):
        """Pack tensorflow dataset."""
        return tf.data.Dataset.from_tensor_slices(
            ({self._const.input_name: snapshots},
            {self._const.output_name_1: labels,
            self._const.output_name_2: snapshots},
            {self._const.output_name_1: prediction_weights,
            self._const.output_name_2: reconstruction_weights})) \
                .shuffle(self.snapshot_cnt) \
                .batch(self._const.batch_size)

    def prepare_groundTruth(
            self,
            reduced_list_var_names,
            dataset):
        # Get bn_snapshots
        snapshots = self.bound_normalize(dataset.snapshots)
        # Get bnr_snapshots
        snapshots = self.reduce(snapshots, reduced_list_var_names)
        # Get bnrg_snapshots
        g_snapshots = self.gridify(snapshots)
        return g_snapshots, dataset.labels, dataset.weights

    def prepare_trimmedGroundTruth(
            self,
            reduced_list_var_names,
            dataset):
        # Get bnrg_snapshots
        g_snapshots, _, _ = \
            self.prepare_groundTruth(reduced_list_var_names, dataset)
        _, pBs, _ = \
            self.approximate(g_snapshots, dataset)
        # Get bnrgt_snapshots, t_labels and t_weights
        g_snapshots, labels, weights = \
            self.trim(pBs, g_snapshots, dataset.labels, dataset.weights)
        return g_snapshots, labels, weights

    def prepare_dataset_from_bn(
            self,
            reduced_list_var_names,
            bn_snapshots,
            dataset):
        # Get bnr_snapshots
        snapshots = self.reduce(bn_snapshots, reduced_list_var_names)
        # Get bnrg_snapshots
        g_snapshots = self.gridify(snapshots)
        _, pBs, _ = self.approximate(g_snapshots, dataset)
#        # Get bnrt_snapshots
#        snapshots, pBs = self.trim(pBs, snapshots, pBs)
        # Get s_pBs
        pBs = self.squeeze(pBs)
        # Get bnr(t)n_snapshots
        snapshots = self.normalize(snapshots)
        ds = self.pack_tf_dataset(
            snapshots=snapshots,
            labels=pBs,
            prediction_weights=np.ones(len(snapshots)),
            reconstruction_weights=np.ones(len(snapshots)))
        minima = np.amin(snapshots, axis=0)
        maxima = np.amax(snapshots, axis=0)
        return ds, minima, maxima, g_snapshots

    def prepare_prediction_plotter(
            self,
            reduced_list_var_names,
            dataset):
        bn_snapshots = self.bound_normalize(dataset.snapshots)
        ds, minima, maxima, g_snapshots = \
            self.prepare_dataset_from_bn(
                reduced_list_var_names, bn_snapshots, dataset)
        corrected_1D = self.correct_1D(g_snapshots)
        corrected_2D = self.correct_2D(g_snapshots)
        return ds, minima, maxima, corrected_1D, corrected_2D

    def prepare_stepper(
            self,
            reduced_list_var_names,
            train_bn_snapshots,
            train_dataset,
            val_bn_snapshots,
            val_dataset):
        train_ds, _, _, _ = \
            self.prepare_dataset_from_bn(
                reduced_list_var_names, train_bn_snapshots, train_dataset)
        val_ds, _, _, _ = \
            self.prepare_dataset_from_bn(
                reduced_list_var_names, val_bn_snapshots, val_dataset)
        return train_ds, val_ds
