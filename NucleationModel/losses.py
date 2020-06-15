import tensorflow as tf

def binary_neg_likelihood(y_actual, y_pred):
	# nB log pB + nA log (1-pB)
	return -(y_actual * tf.math.log(y_pred) \
		+ (1-y_actual) * tf.math.log(1-y_pred))

def binomial_neg_likelihood(y_actual, y_pred):
	# label here symbolizes number of paths that ended in B (0,1,2)
	return -(2*y_actual * tf.math.log(y_pred) \
		+ (2*(1-y_actual)) * tf.math.log(1-y_pred))

def log_loss(y_actual, y_pred):
	return tf.math.log(abs(y_actual-y_pred))

def difference_of_logs(y_actual, y_pred):
	#return tf.math.log(y_actual)
	return abs(tf.math.log(y_actual)-tf.math.log(y_pred))

def KL_divergence(y_actual, y_pred):
	return y_actual * tf.math.log(abs(y_actual/y_pred))

def KL_divergence_inv(y_actual, y_pred):
	return y_pred * tf.math.log(abs(y_pred/y_actual))