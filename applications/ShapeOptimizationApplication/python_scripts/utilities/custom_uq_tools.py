import numpy as np
import chaospy as cp

def generate_samples(distribution, num_samples=1, strategy="random"):
    """
    Generate samples from a given distribution using the specified sampling strategy.

    Parameters:
        distribution: chaospy.Distribution
            The distribution to sample from.
        num_samples: int
            The number of samples to generate.
        strategy: str
            The sampling strategy to use ('random', 'latin_hypercube', 'halton', 'hammersley').

    Returns:
        np.ndarray
            Generated samples.
    """
    # Sampling strategies using chaospy
    if strategy == "random":
        samples = distribution.sample(size=num_samples, rule="random").T
    elif strategy == "latin_hypercube":
        samples = distribution.sample(size=num_samples, rule="latin_hypercube").T
    elif strategy == "halton":
        samples = distribution.sample(size=num_samples, rule="halton").T
    elif strategy == "hammersley":
        samples = distribution.sample(size=num_samples, rule="hammersley").T
    elif strategy == "deterministic":
        samples = np.zeros((1,2))
    else:
        raise ValueError("Unknown sampling strategy: {}".format(strategy))

    return samples


def calculate_force_vectors_z(samples, magnitude=100000):
    """
    Calculate force vectors from the given samples.

    Parameters:
        samples: np.ndarray
            The samples containing angles in degrees.
        magnitude: float
            The magnitude of the force vector.

    Returns:
        np.ndarray
            Calculated force vectors.
    """
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

def calculate_force_vectors_pos_z(samples, magnitude=100000):
    """
    Calculate force vectors from the given samples.

    Parameters:
        samples: np.ndarray
            The samples containing angles in degrees.
        magnitude: float
            The magnitude of the force vector.

    Returns:
        np.ndarray
            Calculated force vectors.
    """
    vectors = []
    for angle_xy_deg, angle_zy_deg in samples:
        # Convert angles to radians
        angle_xy_rad = np.radians(angle_xy_deg)
        angle_zy_rad = np.radians(angle_zy_deg)

        # Calculate the components of the vector
        z_component = 1 * magnitude * np.cos(angle_xy_rad) * np.cos(angle_zy_rad)
        x_component = magnitude * np.sin(angle_xy_rad)
        y_component = magnitude * np.sin(angle_zy_rad)

        vectors.append([x_component, y_component, z_component])

    return np.array(vectors)


import numpy as np

def calculate_force_vectors_x(samples, magnitude=100000):
    """
    Calculate force vectors with a variation around the negative x direction from the given samples.

    Parameters:
        samples: np.ndarray
            The samples containing angles in degrees.
        magnitude: float
            The magnitude of the force vector.

    Returns:
        np.ndarray
            Calculated force vectors.
    """
    vectors = []
    for angle_yz_deg, angle_zx_deg in samples:
        # Convert angles to radians
        angle_yz_rad = np.radians(angle_yz_deg)
        angle_zx_rad = np.radians(angle_zx_deg)

        # Calculate the components of the vector
        x_component = -1 * magnitude * np.cos(angle_yz_rad) * np.cos(angle_zx_rad)
        y_component = magnitude * np.sin(angle_yz_rad)
        z_component = magnitude * np.sin(angle_zx_rad)

        vectors.append([x_component, y_component, z_component])

    return np.array(vectors)


def calculate_force_vectors_xz(samples, magnitude=100000):
    """
    Calculate force vectors with a variation around the diagonal in the negative x and z direction
    from the given samples.

    Parameters:
        samples: np.ndarray
            The samples containing angles in degrees.
        magnitude: float
            The magnitude of the force vector.

    Returns:
        np.ndarray
            Calculated force vectors.
    """
    # Base vector in the negative x and z direction
    base_vector = np.array([-1, 0, -1])
    base_vector = base_vector / np.linalg.norm(base_vector)  # Normalize the base vector

    vectors = []
    for angle_xy_deg, angle_zy_deg in samples:
        # Convert angles to radians
        angle_xy_rad = np.radians(angle_xy_deg)
        angle_zy_rad = np.radians(angle_zy_deg)

        # Calculate the perturbation components based on the angles
        perturb_x = np.sin(angle_xy_rad) * np.cos(angle_zy_rad)
        perturb_y = np.sin(angle_zy_rad)
        perturb_z = np.cos(angle_xy_rad) * np.cos(angle_zy_rad)

        # Perturbation vector
        perturbation_vector = np.array([perturb_x, perturb_y, perturb_z])

        # Force vector is the combination of the base vector and the perturbation, scaled by the magnitude
        force_vector = magnitude * (base_vector + perturbation_vector)

        vectors.append(force_vector)

    return np.array(vectors)

def generate_distribution(distribution_parameters):
    distribution_type = distribution_parameters["type"].GetString()

    if distribution_type == "uniform":
        lower = distribution_parameters["lower"].GetDouble()
        upper = distribution_parameters["upper"].GetDouble()
        distribution = cp.J(cp.Uniform(lower, upper), cp.Uniform(lower, upper))
    elif distribution_type == "truncated_normal":
        mean = distribution_parameters["mean"].GetDouble()
        std_dev = distribution_parameters["std_dev"].GetDouble()
        lower = distribution_parameters["lower"].GetDouble()
        upper = distribution_parameters["upper"].GetDouble()
        distribution = cp.J(cp.TruncNormal(lower, upper,mean,std_dev), cp.TruncNormal(lower, upper,mean,std_dev))
    elif distribution_type == "normal":
        mean = distribution_parameters["mean"].GetDouble()
        std_dev = distribution_parameters["std_dev"].GetDouble()
        distribution = cp.J(cp.Normal(mean,std_dev), cp.Normal(mean,std_dev))
    elif distribution_type == "beta":
        alpha = distribution_parameters["alpha"].GetDouble()
        beta = distribution_parameters["beta"].GetDouble()
        lower = distribution_parameters["lower"].GetDouble()
        upper = distribution_parameters["upper"].GetDouble()
        distribution = cp.J(cp.Beta(alpha, beta, lower, upper), cp.Beta(alpha, beta, lower, upper))
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    return distribution

