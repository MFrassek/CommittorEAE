import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class PlotData():
	def __init__(
			self, train_snapshots, minima, maxima,
			train_labels, train_weights, stamp):
		self._train_snapshots = train_snapshots
		self._minima = minima
		self._maxima = maxima
		self._train_labels = train_labels
		self._train_weights = train_weights
		self._stamp = stamp

	def __str__(self):
		return "Minima: {}\nMaxima: {}\nStamp: {}"\
			.format(self._minima, self._maxima, self._stamp)

	@property
	def stamp(self):
		return self._stamp
	
	def plot_super_map(
			self, subfig_size: int, used_variable_names: list, 
			name_to_list_position: dict, resolution: int, 
			vmin: float, vmax: float, 
			method, model = None, 
			points_of_interest = None, fill_val = 0):
		"""
		params:
			used_variable_names: list
				all values to be visited as x_pos
			used_variable_names: list
				all values to be visited as y_pos
			v_min: float/int
				lower bound on the color map
			v_max: float/int
				upper bound on the color map
			resolution: int
				resolution of the produced output list and corresponding figure
			model:
				tf model used for predictions
				default None
			fill_val: float/int
				value assigned to all dimensions not specifically targeted
				default 0 (mean of the normalized list)
		"""
#		assert len(used_variable_names) > 1, "len(used_variable_names) must exceed 1."
		if model == None:
			suptitle = "Given labels depending on input"
			model_name = "Given"
			out_size = 1
		else:
			suptitle = "Predicted {} depending on {}"\
							.format(model.output_names[0], model.input_names[0])
			model_name = model.name
			out_size = model.layers[-1].output_shape[1]

		super_map = []			
		for i in range(len(used_variable_names)):
			super_map.append([])   
			for j in range(len(used_variable_names)):
				super_map[-1].append([])
				if j < i:
					print("{}: {}\t{}: {}".format(
						i, used_variable_names[i], 
						j, used_variable_names[j]))
					label_map = method(
						x_pos = name_to_list_position[used_variable_names[i]], 
						y_pos = name_to_list_position[used_variable_names[j]],
						resolution = resolution,
						model = model,
						points_of_interest = points_of_interest,
						fill_val = fill_val)
					super_map[-1][-1].append(label_map)
		for k in range(out_size):
			print(k)
			fig, axs = plt.subplots(
				len(used_variable_names), len(used_variable_names),
				figsize = (
					subfig_size * len(used_variable_names), 
					subfig_size * len(used_variable_names)))
			fig.suptitle(
				suptitle, 
				fontsize = subfig_size*len(used_variable_names)*2)				
			
			for i in range(len(used_variable_names)):
				for j in range(len(used_variable_names)):
					# Defines new_axs to take care of different
					# handling of only one column of subplots.
					if len(used_variable_names) == 1:
						new_axs = axs[i]
					else:
						new_axs = axs[i][j]
					new_axs.tick_params(
						axis='both',
						which='both',
						bottom=False,
						top=False,
						labelbottom=False,
						left = False,
						labelleft= False)  

					if j < i:
						if vmin >= 0:
							if False:
								im = new_axs.imshow(
									super_map[i][j][0][k][::-1], 
									cmap='coolwarm', 
									interpolation='nearest', 
									vmin=vmin, 
									vmax=vmax)
							else:
								im = new_axs.imshow(
									super_map[i][j][0][k][::-1], 
									cmap='coolwarm', 
									interpolation='nearest',
									norm=mpl.colors.LogNorm(
										vmin=0.01, 
										vmax=vmax))
						else:
							im = new_axs.imshow(
								super_map[i][j][0][k][::-1], 
								cmap='coolwarm', 
								interpolation='nearest', 
								norm=mpl.colors.SymLogNorm(
									linthresh=0.01*(vmax-vmin), 
									linscale=0.1*(vmax-vmin), 
									vmin=vmin, vmax=vmax))
						# Only sets the leftmost and lowest label.
						if i == len(used_variable_names) - 1:
							new_axs.set_xlabel(
								"${}$".format(used_variable_names[j]),
								fontsize=subfig_size * 10)
						if j == 0:
							new_axs.set_ylabel(
								"${}$".format(used_variable_names[i]),
								fontsize=subfig_size * 10)
						# Overwrites labels if predictions are based on the bn.
						if model != None:
							if model.input_names[0] == "encoded_snapshots":
								if i == len(used_variable_names) - 1:
									new_axs.set_xlabel(
										"b{}".format(used_variable_names[j]),
										fontsize=subfig_size * 10)
								if j == 0:
									new_axs.set_ylabel(
										"b{}".format(used_variable_names[i]),
										fontsize=subfig_size * 10)
					else:
						# Remove all subplots where i >= j.
						new_axs.axis("off")
			cax,kw = mpl.colorbar.make_axes([ax for ax in axs])
			cbar = plt.colorbar(im, cax=cax, **kw)
			cbar.ax.tick_params(labelsize=subfig_size * len(used_variable_names))
			plt.savefig("results/{}_fv{}_outN{}_r{}_{}_p{}_map.png"\
				.format(self._stamp, fill_val, k, resolution, model_name[0],
					str(points_of_interest != None)[0]))
			plt.show()
		return super_map

	def calc_map_given(
			self, x_pos, y_pos, resolution,
			model = None, points_of_interest = None, fill_val = 0):
		label_map = [[0 for y in range(resolution)] for x in range(resolution)]
		weight_map = [[0 for y in range(resolution)] for x in range(resolution)]		
		for nr in range(len(self._train_snapshots)):
			x_int = int(self._train_snapshots[nr][x_pos])
			y_int = int(self._train_snapshots[nr][y_pos])
			if x_int >= 0 and x_int <= resolution-1 and y_int >= 0 \
					and y_int <= resolution-1:
				label_map[x_int][y_int] = label_map[x_int][y_int] \
					+ self._train_labels[nr] \
					* self._train_weights[nr]
				weight_map[x_int][y_int] = weight_map[x_int][y_int] \
					+ self._train_weights[nr]
		#print(np.array(label_map))
		label_map = [[label_map[i][j] / weight_map[i][j] \
			if weight_map[i][j] > 0 else float("NaN") \
			for j in range(len(label_map[i]))] \
			for i in range(len(label_map))]
		return np.array([label_map])

	def calc_partial_map_given(
			self, x_pos, y_pos, resolution,
			model = None, points_of_interest = None, fill_val = 0):
		xys = list(set([(int(ele[x_pos]),int(ele[y_pos])) \
			for ele in points_of_interest]))
		label_map = self.calc_map_given(
			x_pos, y_pos, 
			resolution, 
			fill_val = fill_val)
		partial_out_map = [[label_map[0][x][y] \
			if (x,y) in xys else float("NaN") \
			for y in range(resolution)] \
			for x in range(resolution)]
		return np.array([partial_out_map])

	def calc_map_generated(
			self, x_pos, y_pos, resolution,
			model = None, points_of_interest = None, fill_val = 0):
		"""
		Makes predictions over the full (normalized) range of 
		two input variables with all other variables fixed to a specific value.
		Outputs the predictions in the form of a list of lists for plotting.
		params:
			model:
				tf model used for predictions
			resolution: int
				resolution of the produced output list and corresponding figure
			x_pos: int
				index of the variable projected on the x-axis
			y_pos: int
				index of the variable projected on the y-axis
			fill_val: float/int
				value assigned to all dimensions not specifically targeted. 
				default 0 (mean of the normalized list)
		"""
		assert x_pos != y_pos, "x_pos and y_pos need to differ"
		in_size = model.layers[0].output_shape[0][1]
		out_size = model.layers[-1].output_shape[1]
		xs = np.linspace(self._minima[x_pos], self._maxima[x_pos], resolution)
		ys = np.linspace(self._minima[y_pos], self._maxima[y_pos], resolution)
		out_map = [[] for i in range(out_size)]
		for x in xs:
			out_current_row = [[] for i in range(out_size)]
			for y in ys:
				# make predicition for current grid point
				prediction = self.calc_map_point(
					model, x, y, x_pos, y_pos, 
					in_size, fill_val)
				#if prediction > 0.5:
				#	print([[x if x_pos == pos_nr else y if \
				#	y_pos == pos_nr else fill_val \
				#	for pos_nr in range(in_size)]])
				for i in range(out_size):
					out_current_row[i].append(prediction[i])
			for i in range(out_size):
				out_map[i].append(out_current_row[i])
		return np.array(out_map)

	def calc_partial_map_generated(
			self, x_pos, y_pos, resolution,
			model = None, points_of_interest = None, fill_val = 0):
		assert x_pos != y_pos, "x_pos and y_pos need to differ"
		in_size = model.layers[0].output_shape[0][1]
		out_size = model.layers[-1].output_shape[1]
		xys = list(set([(int(ele[x_pos]),int(ele[y_pos])) \
			for ele in points_of_interest]))
		#ys = list(set([ele[y_pos] for ele in points_of_interest]))
		#print(xys)
		#print(len(xys))
		#print(ys)
		out_map = [[[float("NaN") for i in range(resolution)] \
			for j in range(resolution)] \
			for k in range(out_size)] 
		xs = np.linspace(self._minima[x_pos], self._maxima[x_pos], resolution)
		ys = np.linspace(self._minima[y_pos], self._maxima[y_pos], resolution)
		for x,y in xys:
			#print(x, y)
			prediction = self.calc_map_point(
				model, xs[x], ys[y], x_pos, y_pos, 
				in_size, fill_val)
			for i in range(out_size):
				out_map[i][x][y] = prediction[i]
		return np.array(out_map)

	def calc_map_point(
			self, model, x, y, x_pos, y_pos, 
			in_size, fill_val = 0):
		return model.predict([[x if x_pos == pos_nr else y if \
					y_pos == pos_nr else fill_val \
					for pos_nr in range(in_size)]])[0]


	def plot_super_scatter(
			self, subfig_size: int, used_variable_names: list, 
			name_to_list_position: dict, resolution: int, 
			model, max_row_len = 6, fill_val = 0):
		"""Generates a superfigure of scater plots.
		Iterates over the different dimensions and based on 
		different input values for one dimensions
		as well as a fixed value fr all other dimensions, 
		predicts the reconstructed value for that dimension.
		An optimal encoding and decoding will yield a diagonal 
		line for each dimension indifferent of the value
		chosen for the other dimensions.
		"""
		suptitle = "Predicted snapshots depending on input"
		row_cnt = ((len(used_variable_names)-1)//max_row_len)+1
		fig, axs = plt.subplots(
			row_cnt, max_row_len,
			figsize=(
				subfig_size*max_row_len,
				subfig_size*row_cnt*1.3))
		#fig, axs = plt.subplots(1, len(used_variable_names), \
					#figsize=(fig_size,fig_size/len(used_variable_names)/0.8))
		fig.suptitle(
			suptitle, 
			fontsize=subfig_size*max_row_len*2, 
			y=1.04 - 0.04*row_cnt)				

		for i in used_variable_names:
			#print(i)   
			#print(name_to_list_position[i])
			xs, ys = self.calc_scatter_generated(
				model = model, 
				x_pos = name_to_list_position[i],
				resolution = resolution, 
				fill_val = fill_val)

			#axs[used_variable_names.index(i)//6][used_variable_names.index(i)%6].tick_params(
			if row_cnt > 1:
				new_axs = axs[(used_variable_names.index(i))//max_row_len]
			else:
				new_axs = axs
			new_axs[used_variable_names.index(i)%max_row_len].tick_params(
				axis='both',
				which='both',
				top=False,
				bottom=False,
				labelbottom=False,
				left = False,
				labelleft= False)	
			im = new_axs[used_variable_names.index(i)%max_row_len]\
				.scatter(xs, ys, s=subfig_size*20)
			new_axs[used_variable_names.index(i)%max_row_len]\
				.set_xlim(
					[self._minima[name_to_list_position[i]],
					self._maxima[name_to_list_position[i]]])
			new_axs[used_variable_names.index(i)%max_row_len]\
				.set_ylim(
					[self._minima[name_to_list_position[i]],
					self._maxima[name_to_list_position[i]]])
			new_axs[used_variable_names.index(i)%max_row_len]\
				.set_xlabel(
					"${}$".format(i),
					fontsize=subfig_size*10)
		# if not all rows are filled 
		# remove the remaining empty subplots in the last row
		if len(used_variable_names)%max_row_len != 0:
			for i in range(len(used_variable_names)%max_row_len, max_row_len):
				new_axs[i].axis("off")
		plt.tight_layout(rect = [0, 0, 1, 0.8])
		plt.savefig("results/{}_fv{}_r{}_scat.png"\
			.format(self._stamp, fill_val, resolution)) 
		plt.show()
		return

	def calc_scatter_generated(self, model, x_pos, resolution, fill_val = 0):
		in_size = model.layers[0].output_shape[0][1]
		xs = np.linspace(self._minima[x_pos], self._maxima[x_pos], resolution)
		ys = []
		for x in xs:
			prediction = model.predict([[x if x_pos == pos_nr else fill_val \
					for pos_nr in range(in_size)]])[0]
			ys.append(prediction[x_pos])
		return xs, ys 

# 	def calc_map_given(self, x_pos, y_pos, resolution):
# 		""""""
# 		xs = np.linspace(self._minima[x_pos], self._maxima[x_pos], resolution)
# 		ys = np.linspace(self._minima[y_pos], self._maxima[y_pos], resolution)
# 		x_span = self._maxima[x_pos] - self._minima[x_pos]
# 		y_span = self._maxima[y_pos] - self._minima[y_pos]
			

# 		# Generate two lists of lists of zeros to add all labels and weights to.
# 		label_map = [[0 for y in ys] for x in xs]
# 		weight_map = [[0 for y in ys] for x in xs]
# 		# Sort the labels of each snapshot to the corresponding 
# 		# "positions" in the grid (by sorting them in the list).
# 		for nr in range(len(self._train_snapshots)):
# 			x_snap = self._train_snapshots[nr][x_pos]
# 			y_snap = self._train_snapshots[nr][y_pos]
# 			# Uses "int" to be able to use for iteration. 
# 			# Uses "round" to round to closest full number. 
# 			# Uses "i-min_x" to offset to start at 0. 
# 			# Uses "//x_span*(resolution-1)" to rescale and return int.
# 			x_int = int((x_snap - self._minima[x_pos])\
# 				/x_span*(resolution-1)+0.5)
# 			y_int = int((y_snap - self._minima[y_pos])\
# 					/y_span*(resolution-1)+0.5)
# 			# After the gridpoint closest to the snapshot position is determined,
# 			# if the snapshot lies within the bounds
# 			# the snapshot's label is multiplied with its weight 
# 			# and added to that grid point.
# 			# The weight is added to the corresponding postion of the weight map.
# 			if x_int >= 0 and x_int <= resolution-1 and y_int >= 0 \
# 					and y_int <= resolution-1:
				
# 				label_map[x_int][y_int] = label_map[x_int][y_int] \
# 					+ self._train_labels[nr] \
# 					* self._train_weights[nr]
# 				weight_map[x_int][y_int] = weight_map[x_int][y_int] \
# 					+ self._train_weights[nr]
# 		# Calculate the weighted mean of the labels associated with
# 		# each grid point. 
# 		# Divide each entry of the label map with the corresponding
# 		# entry in the weight map. If no snapshots were associated
# 		# with that grid point, and the corresponding weight in the 
# 		# weight_map is 0, set the cell value to NaN. 
# #		label_map = list(map(lambda y: list(map(lambda x: np.mean(x) \
# #						if len(x) > 0 else float('Nan'),y)),label_map))
# 		print(label_map)
# 		print(weight_map)
# 		label_map = [[label_map[i][j] / weight_map[i][j] \
# 			if weight_map[i][j] > 0 else float("NaN") \
# 			for j in range(len(label_map[i]))] \
# 			for i in range(len(label_map))]
# 		return np.array([label_map])

	# def calc_full_map_given(self, resolution):
	# 	dims = range(len(self._minima))
	# 	xs_s = [np.linspace(self._minima[x_pos], self._maxima[x_pos], resolution) for x_pos in dims]
	# 	print(xs)
	# 	x_spans = [self._maxima[x_pos] - self._minima[x_pos] for x_pos in dims]
	# 	print(x_spans)
	# 	rounded_snapshots = []
	# 	for nr in range(len(self._train_snapshots)):
	# 		for dim in dims:
	# 			pass