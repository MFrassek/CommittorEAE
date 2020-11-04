from helperFunctions import \
    make_halfpoint_divided_label_colormap, \
    make_density_colormap
from losses import *
from tensorflow import keras


class Const():
    def __init__(self, dataSetType):
        self._dataSetType = dataSetType
        if dataSetType == "DW" or dataSetType == "ZP":
            self._name_to_list_position = {
                "x_{1}": 0, "x_{2}": 1, "x_{3}": 2, "x_{4}": 3, "x_{5}": 4,
                "x_{6}": 5, "x_{7}": 6, "x_{8}": 7, "x_{9}": 8, "x_{10}": 9}
            self._used_variable_names = [
                "x_{1}", "x_{2}", "x_{3}", "x_{4}", "x_{5}"]
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
                "MCG": 0, "N_{w,4}": 1, "N_{w,3}": 2, "N_{w,2}": 3,
                "N_{sw,3-4}": 4, "N_{sw,2-3}": 5, "F4": 6, "R_g": 7,
                "5^{12}6^{2}": 8, "5^{12}": 9, "CR": 10, "N_{s,2}": 11,
                "N_{s,3}": 12, "N_{c,2}": 13, "N_{c,3}": 14, "N_{s,4}": 15,
                "N_{c,4}": 16, "5^{12}6^{3}": 17, "5^{12}6^{4}": 18,
                "4^{1}5^{10}6^{2}": 19, "4^{1}5^{10}6^{3}": 20,
                "4^{1}5^{10}6^{4}": 21}
            self._used_variable_names = [
                "MCG", "N_{w,4}", "N_{w,3}", "N_{w,2}", "N_{sw,3-4}",
                "N_{sw,2-3}", "F4", "R_g", "5^{12}6^{2}", "5^{12}", "CR",
                "N_{s,2}", "N_{s,3}", "N_{c,2}", "N_{c,3}", "N_{s,4}",
                "N_{c,4}", "5^{12}6^{3}", "5^{12}6^{4}", "4^{1}5^{10}6^{2}",
                "4^{1}5^{10}6^{3}", "4^{1}5^{10}6^{4}"]
            # Name of the folder in which the TIS data is found
            self._TIS_folder_name = "RPE_org"
            self._TIS_highest_interface_name = "mcg100"
            # Name of the folder in which the TPS paths are found
            self._TPS_folder_name = "TPS"
            # MCG threshold below which a snapshot belongs to state A
            self._mcg_A = 18
            # MCG threshold above which a snapshot belongs to state B
            self._mcg_B = 120
            # Fraction of paths used from the read files
            self._used_TIS_frac = 0.1
            self._used_TPS_frac = 0.1

        self._used_name_to_list_position = {
            self._used_variable_names[i]: i
            for i in range(len(self._used_variable_names))}
        # Labels assigned to the four types of paths
        self._AA_label = 0.0
        self._AB_label = 1.0
        self._BA_label = 0.0
        self._BB_label = 1.0
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
        self._bottleneck_size = 1
        # Factor of hidden layer nodes relative to input nodes
        self._node_mult = 4
        # Number ob hidden layers in the encoder
        self._encoder_hidden = 4
        # Number ob hidden layers in the decoder_1
        self._decoder_1_hidden = 4
        # Number ob hidden layers in the decoder_2
        self._decoder_2_hidden = 4
        # Activation function in the encoder
        self._encoder_act_func = "tanh"
        # Activation function in the decoder_1
        self._decoder_1_act_func = "sigmoid"
        # Activation function in the decoder_2
        self._decoder_2_act_func = "tanh"
        # Ratio of weights for label and reconstruction loss
        self._loss_weights = [1, 0.1]
        # Names of input and output in the model.
        self._input_name = "Input"
        self._output_name_1 = "Committor"
        self._output_name_2 = "Reconstruction"
        # List off losses determined by the model.
        self._loss_names = ["total", self._output_name_1, self._output_name_2]
        # Loss functions used for the autoencoder_1
        self._loss_function_1 = binaryNegLikelihood
        # Loss functions used for the autoencoder_2
        self._loss_function_2 = keras.losses.MeanAbsoluteError()
        # Number of epochs used for model training
        self._epochs = 10

        """Visualization parameters"""
        # Resolution for the calc_* and plot_* functions
        self._resolution = 25
        # Sub-figure size for the plot_* functions
        # self._subfig_size = 5
        self._subfig_size = 2
        # Lower bondary for a logarithmic colormap
        self._logvmin = 10**(-4)
        # Colormap used for the heat map plots
        self._label_cmap = make_halfpoint_divided_label_colormap(self._logvmin)
        # Colormap used for the desnity plots
        self._density_cmap = make_density_colormap()
        # Thresholds for correlation between dimensions
        self._corr_thresholds = [0.5, 0.1]
        # List of colors for plt.plots
        self._plt_colors = [
            "c", "g", "r", "indigo", "y", "m",
            "k", "lightpink", "orange", "olive", "b"]

    @property
    def dataSetType(self):
        return self._dataSetType

    @property
    def name_to_list_position(self):
        return self._name_to_list_position

    @property
    def used_variable_names(self):
        return self._used_variable_names

    @property
    def used_name_to_list_position(self):
        return self._used_name_to_list_position

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
    def label_cmap(self):
        return self._label_cmap

    @property
    def density_cmap(self):
        return self._density_cmap

    @property
    def corr_thresholds(self):
        return self._corr_thresholds

    @property
    def plt_colors(self):
        return self._plt_colors

    @property
    def data_stamp(self):
        return "kl{}_oc{}".format(
            "_".join(self._keep_labels), self._outlier_cutoff)

    @property
    def model_stamp(self):
        return "bn{}_{}*({}{}+{}{}|{}{})_lw{}:{}_e{}" \
            .format(
                str(self._bottleneck_size),
                str(self._node_mult),
                str(self._encoder_hidden),
                str(self._encoder_act_func),
                str(self._decoder_1_hidden),
                str(self._decoder_1_act_func),
                str(self._decoder_2_hidden),
                str(self._decoder_2_act_func),
                self._loss_weights[0],
                self._loss_weights[1],
                self._epochs)

    # Define setter methods for all variables that can be changed.
    @used_variable_names.setter
    def used_variable_names(self, x):
        assert isinstance(x, list), "Can only be set to type list"
        self._used_variable_names = x
        self._used_name_to_list_position = {
            self._used_list_var_names[i]: i
            for i in range(len(self._used_list_var_names))}

    @TIS_folder_name.setter
    def TIS_folder_name(self, x):
        assert isinstance(x, str), "Can only be set to type str"
        self._TIS_folder_name = x

    @TPS_folder_name.setter
    def TPS_folder_name(self, x):
        assert isinstance(x, str), "Can only be set to type str"
        self._TPS_folder_name = x

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
