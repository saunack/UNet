import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
# Elastic transform
def elastic_transformations(alpha, sigma, interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(image):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = image.shape
        # Make random fields
        dx = np.random.uniform(-1, 1, image_shape) * alpha
        dy = np.random.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy

        gx = np.random.randn(3,3)*sigma
        gy = np.random.randn(3,3)*sigma

        sdx = convolve2d(dx,gx,mode='same')
        sdy = convolve2d(dy,gy,mode='same')

        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)

        # Map cooordinates from image to distorted index set
        transformed_images = map_coordinates(image, distorted_indices, mode='reflect',
                                              order=interpolation_order).reshape(image_shape)
        return transformed_images
    return _elastic_transform_2D

from PIL import Image
#a = Image.open('UNet/Data/train-volume.tif')
a = Image.open('../Data/grid.jpeg')
b = np.array(a)

#e = elastic_transformations(1,10,interpolation_order=1)(b)
#E = Image.fromarray(e)
#a.show()
#E.show()

alpha = 1
sigma = 50
interpolation_order = 1
image = b
image_shape = image.shape
# Make random fields
dx = np.random.uniform(-1, 1, image_shape) * alpha
dy = np.random.uniform(-1, 1, image_shape) * alpha
# Smooth dx and dy

gx = np.random.randn(3,3)*sigma
gy = np.random.randn(3,3)*sigma

sdx = convolve2d(dx,gx,mode='same')
sdy = convolve2d(dy,gy,mode='same')

# Make meshgrid
x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
# Distort meshgrid indices
distorted_indices = (y + sdy).reshape(-1, 1), \
                    (x + sdx).reshape(-1, 1)

# Map cooordinates from image to distorted index set
t = map_coordinates(image, distorted_indices, mode='reflect',
                                        order=interpolation_order).reshape(image_shape)
def a(x,show=True):
    b = Image.fromarray(x)
    if show:
        b.show()
    return b
