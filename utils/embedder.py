from typing import List
import numpy as np

def embed_to_box(x_coordinates: List[float], y_coordinates: List[float]):
    '''
    Returns an np array with shape (21,2)
    '''
    xs = np.array(x_coordinates)
    ys = np.array(y_coordinates)
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    
    width = x_max - x_min
    height = y_max - y_min
    
    xs = (xs - x_min) / width
    ys = (ys - y_min) / height
    
    return np.column_stack((xs, ys))
    
    
def embed_from_palm_base(x_coordinates: List[float], y_coordinates: List[float]):
    '''
    Returns an np array with shape (20,2)
    
    It is 20 (for 21 landmarks) since the anchor point at base of palm (0.0, 0.0) is omitted
    '''
    xs = np.array(x_coordinates)
    ys = np.array(y_coordinates)
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    
    width = x_max - x_min
    height = y_max - y_min
    
    xs = (xs - xs[0]) / (width)
    ys = (ys - ys[0]) / (height)
    
    return np.column_stack((xs[1:], ys[1:]))