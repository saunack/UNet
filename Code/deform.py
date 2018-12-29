import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from PIL import Image
import torch

# Elastic transform

def elastic(alpha, sigma, interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _get_params(image_shape):
        # Make random fields
        dx = np.random.uniform(-1, 1, image_shape) * alpha
        dy = np.random.uniform(-1, 1, image_shape) * alpha
        return dx, dy
    
    def _deform(image, dx=None, dy=None):
        """`image` is a PIL Image object of shape (M, N) size M*N."""
        image = np.array(image)
        # Take measurements
        image_shape = image.shape

        if dx is None:
          dx = np.random.uniform(-1, 1, image_shape) * alpha
        if dy is None:
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
        return Image.fromarray(transformed_images)
    return _get_params, _deform
