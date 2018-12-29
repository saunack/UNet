import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
import torch
# Elastic transform

def elastic_transformations(alpha, sigma, interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(image):
        """`images` is a tensor of shape (M, N) size M*N."""
        image = image.numpy()
        # Take measurements
        image_shape = image.shape
        # Make random fields
        dx = np.random.uniform(-1, 1, image_shape) * alpha
        dy = np.random.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy

        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')

        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set
        transformed_images = map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
        return torch.tensor(transformed_images)
    return _elastic_transform_2D
