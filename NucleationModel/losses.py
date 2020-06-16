import tensorflow as tf

def binaryNegLikelihood(y_actual, y_pred):
	# nB log pB + nA log (1-pB)
	return -(y_actual * tf.math.log(y_pred) \
		+ (1-y_actual) * tf.math.log(1-y_pred))

def binomialNegLikelihood(y_actual, y_pred):
	# label here symbolizes number of paths that ended in B (0,1,2)
	return -(2*y_actual * tf.math.log(y_pred) \
		+ (2*(1-y_actual)) * tf.math.log(1-y_pred))

def logLoss(y_actual, y_pred):
	return tf.math.log(abs(y_actual-y_pred))

def differenceOfLogs(y_actual, y_pred):
	#return tf.math.log(y_actual)
	return abs(tf.math.log(y_actual)-tf.math.log(y_pred))

def KLDivergence(y_actual, y_pred):
	return y_actual * tf.math.log(abs(y_actual/y_pred))

def KLDivergenceInv(y_actual, y_pred):
	return y_pred * tf.math.log(abs(y_pred/y_actual))

def absoluteError(y_actual, y_pred):
	return abs(y_actual - y_pred)

def logAbsoluteError(y_actual, y_pred):
	return tf.math.log(abs(y_actual - y_pred))

def squaredError(y_actual, y_pred):
	return (y_actual - y_pred)**2

def logSquaredError(y_actual, y_pred):
	return tf.math.log((y_actual - y_pred)**2)