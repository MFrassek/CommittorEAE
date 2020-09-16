from helperFunctions import make_halfpoint_divided_colormap
from losses import *
from tensorflow import keras


class Const():
    def __init__(self, dataSetType):
        self._dataSetType = dataSetType
        if dataSetType == "DW" or dataSetType == "ZP":
            self._name_to_list_position = {
                "x_{1}": 0,
                "x_{2}": 1,
                "x_{3}": 2,
                "x_{4}": 3,
                "x_{5}": 4,
                "x_{6}": 5,
                "x_{7}": 6,
                "x_{8}": 7,
                "x_{9}": 8,
                "x_{10}": 9}
            if dataSetType == "DW":
                # Name of the folder in which the DW data is found
                self._toy_folder_name = "DW"
            elif dataSetType == "ZP":
                # Name of the folder in which the ZP data is found
                self._toy_folder_name = "ZP"
            # Fraction of paths used from the read files
            self._used_toy_frac = 1
        elif dataSetType == "MH":
            self._name_to_list_position = {
                "MCG": 0,
                "N_{w,4}": 1,
                "N_{w,3}": 2,
                "N_{w,2}": 3,
                "N_{sw,3-4}": 4,
                "N_{sw,2-3}": 5,
                "F4": 6,
                "R_g": 7,
                "5^{12}6^{2}": 8,
                "5^{12}": 9,
                "CR": 10,
                "N_{s,2}": 11,
                "N_{s,3}": 12,
                "N_{c,2}": 13,
                "N_{c,3}": 14,
                "N_{s,4}": 15,
                "N_{c,4}": 16,
                "5^{12}6^{3}": 17,
                "5^{12}6^{4}": 18,
                "4^{1}5^{10}6^{2}": 19,
                "4^{1}5^{10}6^{3}": 20,
                "4^{1}5^{10}6^{4}": 21}
            # Name of the folder in which the TIS data is found
            self._TIS_folder_name = "RPE_org"
            self._TIS_highest_interface_name = "mcg100"
            # Name of the folder in which the TPS paths are found
            self._TPS_folder_name = "TPS"
            # MCG threshold below which a snapshot belongs to state A
            self._mcg_A = 18
            # MCG threshold above which a snapshot belongs to state B
            self._mcg_B = 120
            # big cage threshold under which a snapshot belongs to amorphous
            self._big_C = 8
            # Fraction of paths used from the read files
            self._used_TIS_frac = 0.1
            self._used_TPS_frac = 0.1

        # Labels assigned to the four types of paths
        self._AA_label = 0.0
        self._AB_label = 1.0
        self._BA_label = 0.0
        self._BB_label = 1.0
        # Weights assigned to the totality of each of the path types
        self._path_type_weights = [1, 1, 0, 0]
        # If True snapshots of transition paths are assigned labels
        # according to their position within the path
        self._progress = False
        # Precision to which data is rounded
        self._precision = 2
        # List of labels to keep
        self._keep_labels = ["AA", "AB", "BA", "BB"]
        # Ratio of training set compared to the whole dataset
        self._train_ratio = 0.6
        # Ratio of validation set compared to whole dataset
        self._val_ratio = 0.1
        # Fraction of most extreme values that are considered
        # outliers to both sides
        self._outlier_cutoff = 0.02
        # Number of bins to balance the pBs
        self._balance_bins = 10

        """System parameters"""
        # Number of cores used
        self._cores_used = 2

        """Tf-Dataset parameters"""
        # set size of batches
        self._batch_size = 64

        """Model parameters"""
        # Number of bottleneck nodes
        self._bottleneck_size = 2
        # Factor of hidden layer nodes relative to input nodes
        self._node_mult = 4
        # Number ob hidden layers in the encoder
        self._encoder_hidden = 4
        # Number ob hidden layers in the decoder_1
        self._decoder_1_hidden = 4
        # Number ob hidden layers in the decoder_2
        self._decoder_2_hidden = 4
        # Activation function in the encoder
        self._encoder_act_func = "linear"
        # Activation function in the decoder_1
        self._decoder_1_act_func = "sigmoid"
        # Activation function in the decoder_2
        self._decoder_2_act_func = "tanh"
        # Regularizer applied to all hidden layers
        self._regularizer = 0.00001
        # Ratio of weights for label and reconstruction loss
        self._loss_weights = [1, 1]
        # Names of input and output in the model.
        self._input_name = "input_snapshots"
        self._output_name_1 = "label"
        self._output_name_2 = "reconstruction"
        # List off losses determined by the model.
        self._loss_names = ["total", self._output_name_1, self._output_name_2]
        # Loss functions used for the autoencoder_1
        self._loss_function_1 = binaryNegLikelihood
        # Loss functions used for the autoencoder_2
        self._loss_function_2 = keras.losses.MeanAbsoluteError()
        # Number of epochs used for model training
        self._epochs = 5

        """Visualization parameters"""
        # Resolution for the calc_* and plot_* functions
        self._resolution = 25
        # Sub-figure size for the plot_* functions
        # self._subfig_size = 5
        self._subfig_size = 2
        # Lower bondary for a logarithmic colormap
        self._logvmin = 10**(-10)
        # Colormap used for the heat map plots
        self._cmap = make_halfpoint_divided_colormap(self._logvmin)
        # Thresholds for correlation between dimensions
        self._corr_thresholds = [0.5, 0.1]
        if min(self._path_type_weights) >= 0 \
                and max(self._path_type_weights) <= 1 \
                and self._decoder_1_act_func != "sigmoid":
            print("'sigmoid' activation function recommended"
                  + " for label prediction.")
        elif min(self._path_type_weights) >= -1 \
                and min(self._path_type_weights) < 0 \
                and max(self._path_type_weights) <= 1 \
                and self._decoder_1_act_func != "tanh":
            print("'tanh' activation function recommended"
                  + "for label prediction.")

    @property
    def dataSetType(self):
        return self._dataSetType

    @property
    def name_to_list_position(self):
        return self._name_to_list_position

    @property
    def toy_folder_name(self):
        return self._toy_folder_name

    @property
    def TIS_folder_name(self):
        return self._TIS_folder_name

    @property
    def TIS_highest_interface_name(self):
        return self._TIS_highest_interface_name

    @property
    def TPS_folder_name(self):
        return self._TPS_folder_name

    @property
    def mcg_A(self):
        return self._mcg_A

    @property
    def mcg_B(self):
        return self._mcg_B

    @property
    def big_C(self):
        return self._big_C

    @property
    def used_toy_frac(self):
        return self._used_toy_frac

    @property
    def used_TIS_frac(self):
        return self._used_TIS_frac

    @property
    def used_TPS_frac(self):
        return self._used_TPS_frac

    @property
    def AA_label(self):
        return self._AA_label

    @property
    def AB_label(self):
        return self._AB_label

    @property
    def BA_label(self):
        return self._BA_label

    @property
    def BB_label(self):
        return self._BB_label

    @property
    def min_label(self):
        return min(
            self._AA_label,
            self._AB_label,
            self._BA_label,
            self._BB_label)

    @property
    def max_label(self):
        return max(
            self._AA_label,
            self._AB_label,
            self._BA_label,
            self._BB_label)

    @property
    def path_type_weights(self):
        return self._path_type_weights

    @property
    def progress(self):
        return self._progress

    @property
    def precision(self):
        return self._precision

    @property
    def keep_labels(self):
        return self._keep_labels

    @property
    def train_ratio(self):
        assert isinstance(self._train_ratio, float) \
            and self._train_ratio > 0.0, \
            "train_ratio needs to be a float higher than 0.0"
        return self._train_ratio

    @property
    def val_ratio(self):
        assert isinstance(self._val_ratio, float) \
            and self._val_ratio > 0.0, \
            "val_ratio needs to be a float higher than 0.0"
        return self._val_ratio

    @property
    def outlier_cutoff(self):
        return self._outlier_cutoff

    @property
    def balance_bins(self):
        return self._balance_bins

    @property
    def cores_used(self):
        return self._cores_used

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def bottleneck_size(self):
        return self._bottleneck_size

    @property
    def node_mult(self):
        return self._node_mult

    @property
    def encoder_hidden(self):
        return self._encoder_hidden

    @property
    def decoder_1_hidden(self):
        return self._decoder_1_hidden

    @property
    def decoder_2_hidden(self):
        return self._decoder_2_hidden

    @property
    def encoder_act_func(self):
        return self._encoder_act_func

    @property
    def decoder_1_act_func(self):
        return self._decoder_1_act_func

    @property
    def decoder_2_act_func(self):
        return self._decoder_2_act_func

    @property
    def regularizer(self):
        return self._regularizer

    @property
    def loss_weights(self):
        return self._loss_weights

    @property
    def label_loss_weight(self):
        return self._loss_weights[0]

    @property
    def reconstruction_loss_weight(self):
        return self._loss_weights[1]

    @property
    def input_name(self):
        return self._input_name

    @property
    def output_name_1(self):
        return self._output_name_1

    @property
    def output_name_2(self):
        return self._output_name_2

    @property
    def loss_names(self):
        return self._loss_names

    @property
    def loss_type_cnt(self):
        return len(self._loss_names)

    @property
    def loss_function_1(self):
        return self._loss_function_1

    @property
    def loss_function_2(self):
        return self._loss_function_2

    @property
    def epochs(self):
        return self._epochs

    @property
    def resolution(self):
        return self._resolution

    @property
    def subfig_size(self):
        return self._subfig_size

    @property
    def logvmin(self):
        return self._logvmin

    @property
    def cmap(self):
        return self._cmap

    @property
    def corr_thresholds(self):
        return self._corr_thresholds

    @property
    def data_stamp(self):
        return "kl{}_p{}_oc{}"\
            .format(
                "_".join(self._keep_labels),
                str(self._progress)[0],
                self._outlier_cutoff)

    @property
    def model_stamp(self):
        return "bn{}_{}*({}{}+{}{}|{}{})_reg{}_pw{}:{}:{}:{}_lw{}:{}_e{}" \
            .format(
                str(self._bottleneck_size),
                str(self._node_mult),
                str(self._encoder_hidden),
                str(self._encoder_act_func),
                str(self._decoder_1_hidden),
                str(self._decoder_1_act_func),
                str(self._decoder_2_hidden),
                str(self._decoder_2_act_func),
                self._regularizer,
                self._path_type_weights[0],
                self._path_type_weights[1],
                self._path_type_weights[2],
                self.path_type_weights[3],
                self._loss_weights[0],
                self._loss_weights[1],
                self._epochs)

    # Define setter methods for all variables that can be changed.
    @TIS_folder_name.setter
    def TIS_folder_name(self, x):
        assert isinstance(x, str), "Can only be set to type str"
        self._TIS_folder_name = x

    @TPS_folder_name.setter
    def TPS_folder_name(self, x):
        assert isinstance(x, str), "Can only be set to type str"
        self._TPS_folder_name = x

    @path_type_weights.setter
    def path_type_weights(self, x):
        assert isinstance(x, list), "Can only be set to type list"
        self._path_type_weights = x

    @keep_labels.setter
    def keep_labels(self, x):
        assert isinstance(x, list), "Can only be set to type list"
        self._keep_labels = x

    @bottleneck_size.setter
    def bottleneck_size(self, x):
        assert isinstance(x, int), "Can only be set to type int"
        self._bottleneck_size = x

    @loss_weights.setter
    def loss_weights(self, x):
        assert isinstance(x, list), "Can only be set to type list"
        self._loss_weights = x

    @outlier_cutoff.setter
    def outlier_cutoff(self, x):
        self._outlier_cutoff = x

    @epochs.setter
    def epochs(self, x):
        assert isinstance(x, int), "Can only be set to type int"
        self._epochs = x
