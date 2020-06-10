import numpy as np
import tensorflow as tf

class DatasetData():
	def __init__(
			self, 
			train_past_snapshots,
			val_past_snapshots,
			test_past_snapshots,
			train_snapshots,
			val_snapshots,
			test_snapshots,
			train_labels,
			val_labels,
			test_labels,
			train_weights,
			val_weights,
			test_weights,
			outlier_cutoff,
			resolution):
		self._train_past_snapshots = train_past_snapshots
		self._val_past_snapshots = val_past_snapshots
		self._test_past_snapshots = test_past_snapshots
		self._train_snapshots = train_snapshots
		self._val_snapshots = val_snapshots
		self._test_snapshots = test_snapshots			
		self._train_labels = train_labels
		self._val_labels = val_labels
		self._test_labels = test_labels
		self._train_weights = train_weights
		self._val_weights = val_weights
		self._test_weights = test_weights
		self._outlier_cutoff = outlier_cutoff
		self._dimensions = len(train_past_snapshots[0])
		self._resolution = resolution

	def __str__(self):
		return "{} training snapshots\n{} testing snapshots\n\
		{} input dimensions"\
			.format(self.train_size, self.test_size, self._dimensions)

	@property
	def train_size(self):
		return len(self._train_past_snapshots)
	@property
	def val_size(self):
		return len(self._val_past_snapshots)
	@property
	def test_size(self):
		return len(self._test_past_snapshots)	

	@property
	def train_past_snapshots(self):
		return self._train_past_snapshots
	@property
	def val_past_snapshots(self):
		return self._val_past_snapshots
	@property
	def test_past_snapshots(self):
		return self._test_past_snapshots
	@property
	def train_snapshots(self):
		return self._train_snapshots
	@property
	def val_snapshots(self):
		return self._val_snapshots
	@property
	def test_snapshots(self):
		return self._test_snapshots
	@property
	def train_labels(self):
		return self._train_labels
	@property
	def val_labels(self):
		return self._val_labels
	@property
	def test_labels(self):
		return self._test_labels
	@property
	def train_weights(self):
		return self._train_weights
	@property
	def val_weights(self):
		return self._val_weights
	@property
	def test_weights(self):
		return self._test_weights
	@property
	def dimensions(self):
		return self._dimensions
	@property
	def resolution(self):
		return self._resolution
		

	@property
	def lower_bound(self):
		"""Calculate the lower_bound of all dimensions based of 
		the outlier_cutoff.
		To avoid repeated calculation, the result is stored in 
		self._lower_bound for later reusal."""
		try:
			return self._lower_bound
		except:
			self._lower_bound = np.percentile(self._train_past_snapshots, \
									100*self._outlier_cutoff, axis = 0)
			return self._lower_bound
	@property
	def upper_bound(self):
		"""Calculate the upper_bound of all dimensions based of
		 the outlier_cutoff.
		To avoid repeated calculation, the result is stored in 
		self._lower_bound for later reusal."""
		try:
			return self._upper_bound
		except:
			self._upper_bound =  np.percentile(self._train_past_snapshots, \
									100*(1-self._outlier_cutoff), axis = 0)
			return self._upper_bound
			
	def bounded_outliers(self, snapshots):
		"""Set the values of a snapshopt that lie outside of the bounds to that 
		bound while leaving the other values unchanged.
		Initially transpose the snapshots to a column list
		For each column, iterate over all entries and compare them to the 
		lower or upper bound of that column. If they are lower or higher, 
		change to the value of that bound.
		Return the transpose of the column list, thereby yielding 
		the cleaned snapshots.
		"""
		column_list = np.transpose(snapshots)
		column_list = [[min(self.upper_bound[col_nr],\
						max(self.lower_bound[col_nr],entry)) \
						for entry in column_list[col_nr]] \
						for col_nr in range(self.dimensions)]
		return np.transpose(column_list)

	@property
	def train_bounded_past_snapshots(self):
		"""When calculating train_bounded_past_snapshots, check if 
		self._mean, self._std and self._inv_std have been assigned yet. 
		If not calculate them in one call to reduce repeated
		calculation of train_bounded_past_snapshots.
		"""
		print("Get bounded")
		bounded =  self.bounded_outliers(self._train_past_snapshots)
		try:
			self._mean
			self._std
			self._inv_std
		except:
			print("Get mean, std and inv_std")
			self._mean = np.mean(bounded, axis = 0)
			# If the std along one dimension is 0, the normalization 
			# later would return a NaN value, therefore std values of 
			# 0 are set to 1 while all other ones remain unchanged.
			self._std = np.array([element if element != 0 else 1 for element \
				in np.std(bounded, axis = 0)])
			# Calculate the inverse of the standard deviations to use 
			# for the normalization of the data. Since division is 
			# significantly more expensive than multiplication,  
			# calculating the inverse once yields an efficiency boost.
			self._inv_std = np.ones(self._dimensions) / self.std
		print("Got bounded")
		return bounded
			
	@property
	def val_bounded_past_snapshots(self):
		return self.bounded_outliers(self._val_past_snapshots)
			
	@property
	def test_bounded_past_snapshots(self):
		return self.bounded_outliers(self._test_past_snapshots)

# should the output data be bounded?
	@property
	def train_bounded_snapshots(self):
		return self.bounded_outliers(self._train_snapshots)

	@property
	def val_bounded_snapshots(self):
		return self.bounded_outliers(self._val_snapshots)

	@property
	def test_bounded_snapshots(self):
		return self.bounded_outliers(self._test_snapshots)


	@property
	def mean(self):
		try:
			return self._mean
		except:
			self.train_bounded_past_snapshots
			return self._mean
	@property
	def std(self):
		try:
			return self._std
		except:
			self.train_bounded_past_snapshots
			return self._std
	@property
	def inv_std(self):
		try:
			return self._inv_std
		except:
			self.train_bounded_past_snapshots
			return self._inv_std

	def normalize(self, snapshots):
		"""Normalize the list by substracting the mean 
		and multiplying with the inverse of the standard deviation.
		"""
		return (snapshots - self.mean) * self.inv_std

	@property
	def train_norm_past_snapshots(self):
		# """When calculating train_norm_past_snapshots, check if 
		# self._minima and self._maxima have been assigned yet. 
		# If not calculate them in one call to reduce repeated
		# calculation of train_norm_past_snapshots.
		# """
		try:
			return self._train_norm_past_snapshots
		except:
			print("Get normed")
			self._train_norm_past_snapshots = \
				self.normalize(self.train_bounded_past_snapshots)
			return self._train_norm_past_snapshots

	# @property
	# def train_norm_past_snapshots(self):
	# 	# """When calculating train_norm_past_snapshots, check if 
	# 	# self._minima and self._maxima have been assigned yet. 
	# 	# If not calculate them in one call to reduce repeated
	# 	# calculation of train_norm_past_snapshots.
	# 	# """
	# 	print("Get normed")
	# 	try:
	# 		self._minima
	# 		self._maxima
	# 		self._train_norm_past_snapshots
	# 	except:
	# 		print("Get minima and maxima") 
	# 		self._train_norm_past_snapshots = \
	# 			self.normalize(self.train_bounded_past_snapshots)
	# 		self._minima = np.amin(self._train_norm_past_snapshots, axis = 0)
	# 		self._maxima = np.amax(self._train_norm_past_snapshots, axis = 0)
	# 	print("Got normed")	
	# 	return normed

	@property
	def val_norm_past_snapshots(self):
		return self.normalize(self.val_bounded_past_snapshots)

	@property
	def test_norm_past_snapshots(self):
		return self.normalize(self.test_bounded_past_snapshots)
		
# should the output data be normalized?
	@property
	def train_norm_snapshots(self):
		return self.normalize(self.train_bounded_snapshots)

	@property
	def val_norm_snapshots(self):
		return self.normalize(self.val_bounded_snapshots)

	@property
	def test_norm_snapshots(self):
		return self.normalize(self.test_bounded_snapshots)

	@property
	def minima(self):
		try:
			return self._minima
		except:
			self._minima = np.amin(self.train_norm_past_snapshots, axis = 0)
			return self._minima
	@property
	def maxima(self):
		try:
			return self._maxima
		except:
			self._maxima = np.amax(self.train_norm_past_snapshots, axis = 0)
			return self._maxima


	def gridify(self, snapshots):
		"""Take a list of snapshots and round all entries
		to the closest gridpoint.
		The positions of the gridpoints are
		determined by the chosen resolution as well as the minima and
		maxima of the snapshots.
		"""
		# Calculate the inverse of the spans to allow for 
		# multiplication instead of division. 
		# Then multiply with the resolution, to reduce the 
		# number of multiplications later.		
		inv_spans_res = np.array([1 / (self.maxima[i] - self.minima[i]) * \
			(self._resolution - 1) for i in range(self._dimensions)])
		# Use broadcasting to shift (- np.array(minima)) and rescale
		# (* inv_spans_res) the entries in snapshots.
		# Then increase all values by 0.5 and floor them to have an
		# efficient way of rounding.
		grid_snapshots = np.floor((snapshots - self.minima) \
			* inv_spans_res + 0.5)
		return grid_snapshots

	@property
	def train_grid_past_snapshots(self):
		return self.gridify(self.train_norm_past_snapshots)
	
	@property
	def val_grid_past_snapshots(self):
		return self.gridify(self.val_norm_past_snapshots)

	@property
	def test_grid_past_snapshots(self):
		return self.gridify(self.test_norm_past_snapshots)

	@property
	def train_grid_snapshots(self):
		return self.gridify(self.train_norm_snapshots)
	
	@property
	def val_grid_snapshots(self):
		return self.gridify(self.val_norm_snapshots)

	@property
	def test_grid_snapshots(self):
		return self.gridify(self.test_norm_snapshots)		

	def plot_data(self):
		return self.train_grid_past_snapshots, \
			np.zeros(self._dimensions), \
			np.ones(self._dimensions)*(self._resolution - 1), \
			self.train_labels, self.train_weights

	def importance_data(self):
		return self.val_norm_past_snapshots, self.val_norm_snapshots, \
			self.val_labels, self.val_weights

	def stepwise_data(self):
		return self.train_norm_past_snapshots, self.train_norm_snapshots, \
			self.train_labels, self.train_weights, \
			self.val_norm_past_snapshots, self.val_norm_snapshots, \
			self.val_labels, self.val_weights