# Hum banayenge - Noise Add karne ke liye
import numpy as np

def add_gaussian_noise(data_column, epsilon=1.0, delta=1e-5):
    """
    Gaussian Differential Privacy Noise Generator
    """
    col_max = data_column.max()
    col_min = data_column.min()
    sensitivity = col_max - col_min if (col_max - col_min) > 0 else 1.0

    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    noise = np.random.normal(loc=0.0, scale=sigma, size=data_column.shape)

    return data_column + noise