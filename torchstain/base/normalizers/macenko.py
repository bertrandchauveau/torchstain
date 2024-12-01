def MacenkoNormalizer(backend='torch'):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import NumpyMacenkoNormalizer
        return NumpyMacenkoNormalizer()
    elif backend == "torch":
        from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
        return TorchMacenkoNormalizer()
    elif backend == "tensorflow":
        from torchstain.tf.normalizers.macenko import TensorFlowMacenkoNormalizer
        return TensorFlowMacenkoNormalizer()
    elif backend == "cupy":
        from torchstain.cupy.normalizers.macenko import CupyMacenkoNormalizer
        return CupyMacenkoNormalizer()
    else:
        raise Exception(f'Unknown backend {backend}')
