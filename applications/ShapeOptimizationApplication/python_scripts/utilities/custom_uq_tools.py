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


def rotate_vectors_y(vectors, angle_deg):
    """
    Rotate the given vectors by a specified angle around the y-axis.

    Parameters:
        vectors: np.ndarray
            The vectors to be rotated.
        angle_deg: float
            The angle in degrees by which to rotate the vectors.

    Returns:
        np.ndarray
            The rotated vectors.
    """
    angle_rad = np.radians(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Rotation matrix for rotation around the y-axis
    rotation_matrix = np.array([
        [cos_angle, 0, -sin_angle],
        [0, 1, 0],
        [sin_angle, 0, cos_angle]
    ])

    # Apply the rotation matrix to each vector
    rotated_vectors = np.dot(vectors, rotation_matrix.T)

    return rotated_vectors

def calculate_force_vectors_xz(samples, magnitude=100000, rotation_angle=45):
    """
    Calculate force vectors with a variation around the negative x direction from the given samples,
    and rotate the resulting vectors by a specified angle around the y-axis.

    Parameters:
        samples: np.ndarray
            The samples containing angles in degrees.
        magnitude: float
            The magnitude of the force vector.
        rotation_angle: float
            The angle in degrees to rotate the vectors around the y-axis.

    Returns:
        np.ndarray
            The rotated force vectors.
    """
    # Calculate the force vectors
    vectors = calculate_force_vectors_x(samples, magnitude)

    # Rotate the vectors
    rotated_vectors = rotate_vectors_y(vectors, rotation_angle)

    return rotated_vectors

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

