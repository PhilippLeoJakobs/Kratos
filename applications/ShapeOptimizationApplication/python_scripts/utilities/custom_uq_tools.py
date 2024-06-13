import numpy as np
from scipy.stats import qmc

def generate_downward_vector(magnitude=100000, num_samples=1, strategy="random"):
    # Sampling strategies
    if strategy == "random":
        samples = np.random.uniform(-45, 45, size=(num_samples, 2))
    elif strategy == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n=num_samples) * 90 - 45  # Transform to [-45, 45] range
    elif strategy == "halton":
        sampler = qmc.Halton(d=2, scramble=False)
        samples = sampler.random(n=num_samples) * 90 - 45  # Transform to [-45, 45] range
    elif strategy == "hammersley":
        sampler = qmc.Hammersley(d=2, scramble=False)
        samples = sampler.random(n=num_samples) * 90 - 45  # Transform to [-45, 45] range
    else:
        raise ValueError("Unknown sampling strategy: {}".format(strategy))
    
    vectors = []
    for angle_xy_deg, angle_zy_deg in samples:
        # Convert angles to radians
        angle_xy_rad = np.radians(angle_xy_deg)
        angle_zy_rad = np.radians(angle_zy_deg)
        
        # Calculate the components of the vector
        z_component = -1 * magnitude * np.cos(angle_xy_rad) * np.cos(angle_zy_rad)
        x_component = magnitude * np.sin(angle_xy_rad)
        y_component = magnitude * np.sin(angle_zy_rad)
        
        vectors.append([x_component, y_component, z_component])
    
    return np.array(vectors)
