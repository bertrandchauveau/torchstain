import keras

def cov(x):
    """
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    E_x = keras.ops.mean(x, axis=1)
    x = x - E_x[:, None]
    return keras.ops.matmul(x, keras.ops.transpose(x)) / (keras.ops.shape(x)[1] - 1)
