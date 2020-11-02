import tensorflow as tf


def binaryNegLikelihood(y_actual, y_pred):
    return -(y_actual * tf.math.log(y_pred)
             + (1-y_actual) * tf.math.log(1-y_pred))


def binomialNegLikelihood(y_actual, y_pred):
    return -(2*y_actual * tf.math.log(y_pred)
             + (2*(1-y_actual)) * tf.math.log(1-y_pred))
