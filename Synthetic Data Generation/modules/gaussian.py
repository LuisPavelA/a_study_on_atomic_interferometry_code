import math

def gaussian(x_coordinate, y_coordinate, intensity_factor):
    
    exponent = math.exp(-(2*(x_coordinate)**2 + (1/2)*(y_coordinate)**2)/(2 * intensity_factor**2))

    return (1/(2 * math.pi * intensity_factor**2)) * exponent