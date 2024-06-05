import numpy as np

def generate_downward_vector(magnitude=100000):
    # Generate random angles in degrees
    angle_xy_deg = np.random.uniform(-45, 45)
    angle_zy_deg = np.random.uniform(-45, 45)
    
    # Convert angles to radians
    angle_xy_rad = np.radians(angle_xy_deg)
    angle_zy_rad = np.radians(angle_zy_deg)
    
    # Calculate the components of the vector
    y_component = -1 * magnitude * np.cos(angle_xy_rad) * np.cos(angle_zy_rad)
    x_component = magnitude * np.sin(angle_xy_rad)
    z_component = magnitude * np.sin(angle_zy_rad)
    
    return np.array([x_component, y_component, z_component])

# Example usage
vector = generate_downward_vector()
print("Generated vector:", vector)

    

dist = "normal"
parameters=1
print(generate_downward_vector(1000000))
