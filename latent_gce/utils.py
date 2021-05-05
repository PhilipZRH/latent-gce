import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import custom_envs


def nice_colormap(min_val=0.1, max_val=0.9, n=500):
    colormap = plt.get_cmap('jet')
    new_colormap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=colormap.name, a=min_val, b=max_val),
        colormap(np.linspace(min_val, max_val, n)))
    return new_colormap


# Single entry water-filling.
def water_filling(singular_values, power=10, tol=0.001):
    singular_values = np.asarray(singular_values)
    dim = len(singular_values)
    noise = 1 / (singular_values + 1e-6)
    water = noise.min() + power / dim
    for i in range(10000):
        current_power = np.maximum(water - noise, 0).sum()
        if abs(current_power - power) < tol:
            break
        else:
            water += (power - current_power) / dim
    capacity = 0.5 * np.log2(np.maximum(water - noise, 0) / noise + 1).sum()
    return capacity


# Batched water-filling.
def batch_water_filling(singular_values, power=10, tol=0.001):
    singular_values = np.asarray(singular_values)
    dim = singular_values.shape[1]
    noise = 1 / (singular_values + 1e-6)
    water = noise.min(axis=1, keepdims=True) + power / dim
    for i in range(10000):
        current_power = np.maximum(water - noise, 0).sum(axis=1, keepdims=True)
        if np.abs(current_power - power).max() < tol:
            break
        else:
            water += (power - current_power) / dim
    capacity = 0.5 * np.log2(np.maximum(water - noise, 0) / noise + 1).sum(axis=1)
    return capacity
