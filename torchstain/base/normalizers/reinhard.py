def ReinhardNormalizer(backend='numpy', method=None):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import NumpyReinhardNormalizer
        return NumpyReinhardNormalizer(method=method)
    elif backend == "torch":
        from torchstain.torch.normalizers import TorchReinhardNormalizer
        return TorchReinhardNormalizer(method=method)
    elif backend == "tensorflow":
        from torchstain.tf.normalizers import TensorFlowReinhardNormalizer
        return TensorFlowReinhardNormalizer(method=method)
    elif backend == "cupy":
        from torchstain.cupy.normalizers import CupyReinhardNormalizer
        return CupyReinhardNormalizer(method=method)
    elif backend == "keras":
        from torchstain.keras.normalizers import KerasReinhardNormalizer
        return KerasReinhardNormalizer(method=method)
    else:
        raise Exception(f'Unknown backend {backend}')
