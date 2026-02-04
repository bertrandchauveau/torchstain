def MultiMacenkoNormalizer(backend="torch", **kwargs):
    if backend == "numpy":
        from torchstain.numpy.normalizers import NumpyMultiMacenkoNormalizer
        return NumpyMultiMacenkoNormalizer(**kwargs)
    if backend == "cupy":
        from torchstain.cupy.normalizers import CupyMultiMacenkoNormalizer
        return CupyMultiMacenkoNormalizer(**kwargs)
    elif backend == "torch":
        from torchstain.torch.normalizers import TorchMultiMacenkoNormalizer
        return TorchMultiMacenkoNormalizer(**kwargs)
    elif backend == "tensorflow":
        from torchstain.tf.normalizers import TensorFlowMultiMacenkoNormalizer
        return TensorFlowMultiMacenkoNormalizer(**kwargs)
    else:
        raise Exception(f"Unsupported backend {backend}")
