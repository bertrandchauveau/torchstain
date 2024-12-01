import cupy as cp

def get_mean_std(I):
    return cp.mean(I), cp.std(I)

def standardize(x, mu, std):
    return (x - mu) / std
