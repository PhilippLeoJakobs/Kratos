import numpy as np

def draw_force_sample(distribution_string, parameter1, parameter2):
    if distribution_string=="normal":
        return np.random.normal(parameter1,parameter2)
    elif distribution_string=="uniform":
        return np.random.uniform(parameters)
    

dist = "normal"
parameters=1
print(draw_force_sample(dist,parameters,parameters))
