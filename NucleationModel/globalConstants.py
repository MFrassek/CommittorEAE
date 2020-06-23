from helperFunctions import make_halfpoint_divided_colormap

class Const():
	def __init__(self):
		# Complete list of variables in the dataset to chose from
		self._all_var_names = {
			0: "MCG",
			1: "N_{w,4}",
			2: "N_{w,3}",
			3: "N_{w,2}",
			4: "N_{sw,3-4}",
			5: "N_{sw,2-3}",
			6: "F4",
			7: "R_g",
			8: "5^{12}6^{2}",
			9: "5^{12}",
			10: "CR",
			11: "N_{s,2}",
			12: "N_{s,3}",
			13: "N_{c,2}",
			14: "N_{c,3}",
			15: "N_{s,4}",
			16: "N_{c,4}",
			17: "5^{12}6^{3}",
			18: "5^{12}6^{4}",
			19: "4^{1}5^{10}6^{2}",
			20: "4^{1}5^{10}6^{3}",
			21: "4^{1}5^{10}6^{4}"
			}

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
			"4^{1}5^{10}6^{4}": 21
			}
		# Ordering of the variables in _all_var_names
		self._all_var_order = [
			0, 11, 12, 15, 13, 14, 16, 
			3, 2, 1, 5, 4, 9, 8, 17, 
			18, 19, 20, 21, 10, 7, 6
			]
		self._all_var_order = [
			0, 11, 12, 15, 13, 14, 16, 
			3, 2, 1, 5, 4, 9, 8, 10, 7, 6
			]
		self._names_in_order = [self._all_var_names[i] \
			for i in self._all_var_order]

		"""Pre-Dataset parameters"""
		# Name of the folder in which the complete RPE data is found
		self._RPE_folder_name = "RPE_mod"
		self._RPE_folder_name = "RPE_org"
		# Name of the folder in which the TPS paths are found
		self._TPS_folder_name = "TPS_mod"
		self._TPS_folder_name = "TPS"
		# MCG threshold below which a snapshot belongs to state A
		self._mcg_A = 18
		# MCG threshold above which a snapshot belongs to state B
		self._mcg_B = 120
		# big cage threshold under which a snapshot belongs to amorphous
		self._big_C = 8
		# Fraction of paths used from the read files
		self._used_frac = 0.1
		# Labels assigned to the four types of paths: AA, AB, BA, BB
		self._path_type_labels  = [0.0, 1.0, 0.0, 0.0]
		# Weights assigned to the totality of each of the path types
		self._path_type_weights = [1, 1, 0, 0]
		# Offset to generate datasets with future predictions
		self._offset = 0
		# If True snapshots of transition paths are assigned labels 
		# according to their position within the path 
		self._progress = False
		# If True transition paths (AB, BA) are taken into account for the dataset
		self._transitioned = True
		# If True paths returning to their starting state (AA, BB) 
		# are taken into account for the dataset
		self._turnedback = True
		# Ratio of training set compared to the whole dataset
		self._train_ratio = 0.6
		# Ratio of validation set compared to whole dataset
		self._val_ratio = 0.1
		# Fraction of most extreme values that are considered outliers to both sides
		self._outlier_cutoff = 0.01
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
		self._encoder_act_func = None
		# Activation function in the decoder_1
		self._decoder_1_act_func = "sigmoid"
		# Activation function in the decoder_2
		self._decoder_2_act_func = "tanh"
		# Ratio of weights for label and reconstruction loss 
		self._loss_weights = [1, 1]
		# Names of input and output in the model.
		self._input_name = "input_snapshots"
		self._output_name_1 = "label"
		self._output_name_2 = "reconstruction"
		# List off losses determined by the model.
		self._loss_names = ["total", self._output_name_1, self._output_name_2]
		# Number of epochs used for model training
		self._epochs = 5

		"""Visualization parameters"""
		# Resolution for the calc_* and plot_* functions
		self._resolution = 15
		# Sub-figure size for the plot_* functions
#		self._subfig_size = 5
		self._subfig_size = 2
		# Lower bondary for a logarithmic colormap
		self._logvmin = 10**(-2)
		# Colormap used for the heat map plots
		self._cmap = make_halfpoint_divided_colormap(self._logvmin)
		# Thresholds for correlation between dimensions
		self._corr_thresholds = [0.5, 0.1]
		if min(self._path_type_weights) >= 0 \
				and max(self._path_type_weights) <= 1 \
				and self._decoder_1_act_func != "sigmoid":
			print("'sigmoid' activation function recommended" \
				+ " for label prediction.")
		elif min(self._path_type_weights) >= -1 \
				and min(self._path_type_weights) < 0 \
				and max(self._path_type_weights) <= 1 \
				and self._decoder_1_act_func != "tanh":
			print("'tanh' activation function recommended" \
				+ "for label prediction.")

	@property
	def name_to_list_position(self):
		return self._name_to_list_position
	

	@property
	def all_var_names(self):
		return self._all_var_names
	@property
	def all_var_order(self):
		return self._all_var_order
	@property
	def names_in_order(self):
		return self._names_in_order
	
	@property
	def var_names(self):
		return self._var_names
	@property
	def var_order(self):
		return self._var_order
	@property
	def RPE_folder_name(self):
		return self._RPE_folder_name
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
	def used_frac(self):
		return self._used_frac
	@property
	def path_type_labels (self):
		return self._path_type_labels 
	@property
	def min_label(self):
		return min(self._path_type_labels)
	@property
	def max_label(self):
		return max(self._path_type_labels)
	@property
	def path_type_weights(self):
		return self._path_type_weights
	@property
	def offset(self):
		return self._offset
	@property
	def progress(self):
		return self._progress
	@property
	def transitioned(self):
		return self._transitioned
	@property
	def turnedback(self):
		return self._turnedback
	@property
	def train_ratio(self):
		return self._train_ratio
	@property
	def val_ratio(self):
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
	def stamp(self):
		return "tr{}_re{}_p{}_o{}_oc{}_bn{}_{}*({}{}+{}{}|{}{})_pw{}:{}:{}:{}_lw{}:{}_e{}" \
			.format(
				str(self._transitioned)[0],str(self._turnedback)[0], 
				str(self._progress)[0], self._offset, self._outlier_cutoff,
				str(self._bottleneck_size), str(self._node_mult), 
				str(self._encoder_hidden), str(self._encoder_act_func), 
				str(self._decoder_1_hidden), str(self._decoder_1_act_func), 
				str(self._decoder_2_hidden), str(self._decoder_2_act_func), 
				self._path_type_weights[0], self._path_type_weights[1], 
				self._path_type_weights[2], self.path_type_weights[3], 
				self._loss_weights[0], self._loss_weights[1], 
				self._epochs)
	
	# Define setter methods for all variables that can be changed.
	@RPE_folder_name.setter
	def RPE_folder_name(self, x):
		assert isinstance(x, str), "Can only be set to type str"
		self._RPE_folder_name = x
	
	@TPS_folder_name.setter
	def TPS_folder_name(self, x):
		assert isinstance(x, str), "Can only be set to type str"
		self._TPS_folder_name = x

	@path_type_weights.setter
	def path_type_weights(self, x):
		assert isinstance(x, list), "Can only be set to type list"
		self._path_type_weights = x

	@offset.setter
	def offset(self, x):
		assert isinstance(x, int), "Can only be set to type int"
		self._offset = x

	@transitioned.setter
	def transitioned(self, x):
		assert isinstance(x, bool), "Can only be set to type bool"
		self._transitioned = x

	@turnedback.setter
	def turnedback(self, x):
		assert isinstance(x, bool), "Can only be set to type bool"
		self._turnedback = x

	@bottleneck_size.setter
	def bottleneck_size(self, x):
		assert isinstance(x, int), "Can only be set to type int"
		self._bottleneck_size = x

	@loss_weights.setter
	def loss_weights(self, x):
		assert isinstance(x, list), "Can only be set to type list"
		self._loss_weights = x

	@epochs.setter
	def epochs(self, x):
		assert isinstance(x, int), "Can only be set to type int"
		self._epochs = x

	def used_names(self, used_vars):
		# Generate a dictionary containing the names of the used variables.
		var_names = {i:self._all_var_names[used_vars[i]] \
			for i in range(len(used_vars))}
		return var_names

	def used_order(self, used_vars):
		# Generate a list containing the order for only the used variables.
		var_order = [used_vars.index(i) \
			for i in self._all_var_order if i in used_vars]
		return var_order
