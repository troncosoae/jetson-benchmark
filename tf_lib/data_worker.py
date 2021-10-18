import tensorflow as tf


def suit4tf(X, Y):
    X_norm = X/255
    X_tf = tf.convert_to_tensor(X_norm, dtype=tf.float32)
    Y_tf = tf.convert_to_tensor(Y, dtype=tf.float32)
    return X_tf, Y_tf
