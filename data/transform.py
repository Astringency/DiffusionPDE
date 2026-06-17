import numpy as np



class PDEtransform:
    """
    Min-Max Normalization to [0, 1].
    input: data.shape = [N, C, H, W].
    """
    def __init__(self, data):
        self.data = data

        self.min = self.data.amin(dim=(0, 2, 3), keepdim=True)
        self.max = self.data.amax(dim=(0, 2, 3), keepdim=True)

        self.transform = self._transform_func
        self.inverse_transform = self._inverse_transform_func

    def _transform_func(self):
        return (self.data - self.min) / (self.max - self.min + 1e-8)
        
    def _inverse_transform_func(self):
        return self.data * (self.max - self.min + 1e-8) + self.min


