import numpy as np
import keras
from torchstain.keras.utils.rgb2lab import _rgb2xyz

_xyz2rgb = keras.ops.inv(_rgb2xyz)

"""
Implementation is based on:
https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""
def lab2rgb(lab):
    lab = keras.ops.cast(lab, dtype="float32")
    # first rescale back from OpenCV format
    lab[..., 0] /= 2.55
    lab[..., 1] -= 128
    lab[..., 2] -= 128

    # convert LAB -> XYZ color domain
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    out = keras.ops.stack([x, y, z], axis=-1)

    mask = out > 0.2068966
    out[mask] = keras.ops.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    out *= keras.ops.convert_to_tensor(np.array((0.95047, 1., 1.08883), dtype=out.dtype))
    
    # convert XYZ -> RGB color domain
    arr = keras.ops.copy(out)
    #arr = keras.ops.dot(arr, _xyz2rgb.T)  #torch.dot does not support 2D tensors
    #https://discuss.pytorch.org/t/how-to-operate-torch-dot-in-matrix-consist-of-vectors-in-pytorch/163134/2
    arr = keras.ops.matmul(arr, _xyz2rgb.T)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * keras.ops.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    return keras.ops.clip(arr, 0, 1)
