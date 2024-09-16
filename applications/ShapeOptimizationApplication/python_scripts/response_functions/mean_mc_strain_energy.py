import KratosMultiphysics
from KratosMultiphysics import Parameters, Logger
from KratosMultiphysics.response_functions.response_function_interface import ResponseFunctionInterface
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.ShapeOptimizationApplication.utilities.custom_uq_tools import generate_samples, calculate_force_vectors_x,calculate_force_vectors_xz,calculate_force_vectors_pos_z,calculate_force_vectors_z, generate_distribution
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import chaospy as cp
import numpy as np
import time as timer
from .uq_strain_energy_response import UQStrainEnergyResponseFunction, ModifyPointLoads, ModifySurfaceLoads

class MeanMCStrainEnergyResponseFunction(UQStrainEnergyResponseFunction):
    def __init__(self, identifier, response_settings, model):
        super().__init__(identifier, response_settings, model)
        self.sampling_strategy = response_settings["sampling_strategy"].GetString()
        self.num_samples = response_settings["num_samples"].GetInt()
        self.extra_samples = response_settings["extra_samples"].GetInt()
        self.load_name = response_settings.Has("load_name") and response_settings["load_name"].GetString() or "PointLoad3D_load"  # Default to "PointLoad3D_load"
        self.force_direction = response_settings.Has("force_direction") and response_settings["force_direction"].GetString() or "z"
        self.load_type = response_settings.Has("load_type") and response_settings["load_type"].GetString() or "PointLoad"
        self.load_magnitude = response_settings.Has("load_magnitude") and response_settings["load_magnitude"].GetInt() or 100000
        self.csv_filename = response_settings["response_type"].GetString()+response_settings["load_name"].GetString()+"+values.csv"  # Define your CSV file name here

    def CalculateValue(self):
        Logger.PrintInfo("StrainEnergyResponse", "Starting primal analysis for response", self.identifier, "Strategy is:", self.sampling_strategy)
        startTime = timer.time()
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for solving the primal analysis", round(timer.time() - startTime, 2), "s")
        startTime = timer.time()

        distribution = generate_distribution(self.distribution_parameters)
        sample_angles = generate_samples(distribution, self.num_samples, self.sampling_strategy)
                # Choose the appropriate force vector calculation function
        # Choose the appropriate force vector calculation function
        if self.force_direction.lower() == 'x':
            samples = calculate_force_vectors_x(sample_angles,  self.load_magnitude)
        elif self.force_direction.lower() == 'z':
            samples = calculate_force_vectors_z(sample_angles,  self.load_magnitude)
        elif self.force_direction.lower() == '+z':
            samples = calculate_force_vectors_pos_z(sample_angles, self.load_magnitude)
        elif self.force_direction.lower() == 'xz':
            samples = calculate_force_vectors_xz(sample_angles,  self.load_magnitude)
        else:
            raise ValueError(f"Unknown force direction: {self.force_direction}")

        sample_value = np.zeros(self.num_samples)
        sample_gradient = [{} for _ in range(self.num_samples)]

        for i in range(self.num_samples):
            x_val = samples[i]
            Logger.PrintInfo(x_val)
            if self.load_type == "PointLoad":
                ModifyPointLoads(self.primal_model_part, x_val,self.load_name)
            elif self.load_type == "SurfaceLoad":
                ModifySurfaceLoads(self.primal_model_part, x_val,self.load_name)
            else:
                raise ValueError(f"Unknown load_type: {self.load_type}")
            self.primal_analysis._GetSolver().Predict()
            self.primal_analysis._GetSolver().SolveSolutionStep()
            self.response_function_utility.Initialize()
            sample_value[i] = self.response_function_utility.CalculateValue()
            self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = sample_value[i]
            self.response_function_utility.CalculateGradient()

            for node in self.primal_model_part.Nodes:
                sample_gradient[i][node.Id] = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)

        value = np.mean(sample_value, dtype=float)
        Logger.PrintInfo(value)
        mean_gradient = {}

        for grad_dict in sample_gradient:
            for node_id, gradient in grad_dict.items():
                if node_id not in mean_gradient:
                    mean_gradient[node_id] = np.zeros_like(gradient)
                mean_gradient[node_id] += gradient

        for node_id in mean_gradient:
            mean_gradient[node_id] /= self.num_samples

        gradient = mean_gradient

        for node in self.primal_model_part.Nodes:
            node.SetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY, gradient[node.Id])

        self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = value
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating the response value", round(timer.time() - startTime, 2), "s")

    def CalculateGradient(self):
        Logger.PrintInfo("StrainEnergyResponse", "Starting gradient calculation for response", self.identifier)
        startTime = timer.time()
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating gradients", round(timer.time() - startTime, 2), "s")
