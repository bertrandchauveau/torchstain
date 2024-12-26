import keras

def get_mean_std(I):
    return keras.ops.mean(I), keras.ops.std(I)

def standardize(x, mu, std):
    return (x - mu) / std
