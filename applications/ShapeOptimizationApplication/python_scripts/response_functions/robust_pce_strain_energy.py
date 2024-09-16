import KratosMultiphysics
from KratosMultiphysics import Parameters, Logger
from KratosMultiphysics.response_functions.response_function_interface import ResponseFunctionInterface
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.ShapeOptimizationApplication.utilities.custom_uq_tools import generate_samples, calculate_force_vectors_x, calculate_force_vectors_pos_z, calculate_force_vectors_z, calculate_force_vectors_xz,generate_distribution
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import chaospy as cp
import numpy as np
import time as timer
from .uq_strain_energy_response import UQStrainEnergyResponseFunction, ModifyPointLoads, ModifySurfaceLoads

class RobustPCEStrainEnergyResponseFunction(UQStrainEnergyResponseFunction):
    def __init__(self, identifier, response_settings, model):
        super().__init__(identifier, response_settings, model)
        self.sampling_strategy = response_settings["sampling_strategy"].GetString()
        self.num_samples = response_settings["num_samples"].GetInt()
        self.pce_order = response_settings["pce_order"].GetInt()
        self.extra_samples = response_settings["extra_samples"].GetInt()
        self.mean_weight = response_settings["mean_weight"].GetDouble()
        self.std_weight = response_settings["std_weight"].GetDouble()
        self.load_name = response_settings.Has("load_name") and response_settings["load_name"].GetString() or "PointLoad3D_load"  # Default to "PointLoad3D_load"
        self.force_direction = response_settings.Has("force_direction") and response_settings["force_direction"].GetString() or "z"
        self.load_type = response_settings.Has("load_type") and response_settings["load_type"].GetString() or "PointLoad"
        self.load_magnitude = response_settings.Has("load_magnitude") and response_settings["load_magnitude"].GetInt() or 100000
        self.csv_filename = "response_values.csv"  # Define your CSV file name here

    def CalculateValue(self):
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Starting primal analysis for response", self.identifier)
        startTime = timer.time()
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for solving the primal analysis", round(timer.time() - startTime, 2), "s")
        startTime = timer.time()

        distribution = generate_distribution(self.distribution_parameters)
        sample_angles = generate_samples(distribution, self.num_samples, self.sampling_strategy)

        if self.force_direction.lower() == 'x':
            sample_force = calculate_force_vectors_x(sample_angles, self.load_magnitude)
        elif self.force_direction.lower() == 'z':
            sample_force = calculate_force_vectors_z(sample_angles, self.load_magnitude)
        elif self.force_direction.lower() == '+z':
            sample_force = calculate_force_vectors_pos_z(sample_angles, self.load_magnitude)
        elif self.force_direction.lower() == 'xz':
            sample_force = calculate_force_vectors_xz(sample_angles, self.load_magnitude)
        else:
            raise ValueError(f"Unknown force direction: {self.force_direction}")

        sample_value = np.zeros(self.num_samples)
        sample_gradient = [{} for _ in range(self.num_samples)]

        for i in range(self.num_samples):

            x_val = sample_force[i]
            Logger.PrintInfo("Sample value: ", x_val)
            if self.load_type == "PointLoad":
                ModifyPointLoads(self.primal_model_part, x_val,self.load_name)
            elif self.load_type == "SurfaceLoad":
                ModifySurfaceLoads(self.primal_model_part, x_val,self.load_name)
            else:
                raise ValueError(f"Unknown load_type: {self.load_type}")
            self.primal_analysis._GetSolver().Initialize()
            self.primal_analysis._GetSolver().Predict()
            self.primal_analysis._GetSolver().SolveSolutionStep()
            self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, self.response_settings)
            self.response_function_utility.Initialize()
            sample_value[i] = self.response_function_utility.CalculateValue()
            self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = sample_value[i]
            self.response_function_utility.CalculateGradient()
            for node in self.primal_model_part.Nodes:
                sample_gradient[i][node.Id] = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)

        samples_array = np.array(sample_angles).T
        poly_expansion = cp.orth_ttr(self.pce_order, distribution)
        pce_model_value = cp.fit_regression(poly_expansion, samples_array, sample_value)
        mean_value = cp.E(pce_model_value, distribution)
        std_dev = np.sqrt(cp.Var(pce_model_value, distribution))
        value = self.mean_weight * mean_value + self.std_weight * std_dev


        gradient_start = timer.time()

        gradient_models = {}
        for node in self.primal_model_part.Nodes:
            node_gradients = np.array([sample_gradient[i][node.Id] for i in range(self.num_samples)])
            node_gradients = node_gradients.reshape(self.num_samples, -1)
            pce_model_gradient = cp.fit_regression(poly_expansion, samples_array, node_gradients)
            gradient_models[node.Id] = pce_model_gradient
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for gradient models", round(timer.time() - startTime, 2), "s")
        robust_gradient = {}
        for node_id, pce_model_gradient in gradient_models.items():
            mean_gradient = cp.E(pce_model_gradient, distribution)

            variance_gradient = np.zeros_like(mean_gradient)
            for i in range(len(mean_gradient)):
                variance_gradient[i] = 2 * cp.E(pce_model_gradient[i] * (pce_model_value - mean_value), distribution)

            std_dev_gradient = variance_gradient / (2 * std_dev)

            robust_gradient[node_id] = self.mean_weight * mean_gradient + self.std_weight * std_dev_gradient

        for node in self.primal_model_part.Nodes:
            node.SetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY, robust_gradient[node.Id])

        self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = value
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for calculating the response value", round(timer.time() - startTime, 2), "s")

        # Save the results to a CSV file
        self._save_results_to_csv(mean_value, std_dev)

    def _save_results_to_csv(self, mean_value, std_dev):
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([mean_value, std_dev])

    def CalculateGradient(self):
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Starting gradient calculation for response", self.identifier)
        startTime = timer.time()
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for calculating gradients", round(timer.time() - startTime, 2), "s")
