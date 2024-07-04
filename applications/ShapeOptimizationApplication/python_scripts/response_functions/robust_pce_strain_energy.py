import KratosMultiphysics
from KratosMultiphysics import Parameters, Logger
from KratosMultiphysics.response_functions.response_function_interface import ResponseFunctionInterface
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.ShapeOptimizationApplication.utilities.custom_uq_tools import generate_samples, calculate_force_vectors, generate_distribution
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import chaospy as cp
import numpy as np
import time as timer
from .uq_strain_energy_response import UQStrainEnergyResponseFunction, ModifyPointLoads
# ==============================================================================
class RobustPCEStrainEnergyResponseFunction(UQStrainEnergyResponseFunction):
    """Linear strain energy response function. It triggers the primal analysis and
    uses the primal analysis results to calculate response value and gradient.

    Attributes
    ----------
    primal_model_part : Model part of the primal analysis object
    primal_analysis : Primal analysis object of the response function
    response_function_utility: Cpp utilities object doing the actual computation of response value and gradient.
    """

    def __init__(self, identifier, response_settings, model):
        super().__init__(identifier, response_settings, model)
        self.sampling_strategy = response_settings["sampling_strategy"].GetString()
        self.num_samples = response_settings["num_samples"].GetInt()
        self.pce_order = response_settings["pce_order"].GetInt()
        self.extra_samples = response_settings["extra_samples"].GetInt()
        self.mean_weight = response_settings["mean_weight"].GetDouble()
        self.std_weight = response_settings["std_weight"].GetDouble()


    def CalculateValue(self):
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Starting primal analysis for response", self.identifier)
        startTime = timer.time()
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for solving the primal analysis", round(timer.time() - startTime, 2), "s")
        startTime = timer.time()
        distribution = generate_distribution(self.distribution_parameters)
        sample_angles = generate_samples(distribution, self.num_samples, self.sampling_strategy)
        sample_force = calculate_force_vectors(sample_angles, magnitude=100000)
        sample_value = np.zeros(self.num_samples)
        sample_gradient = [{} for _ in range(self.num_samples)]

        for i in range(self.num_samples):
            x_val = sample_force[i]
            Logger.PrintInfo("Sample value: ", x_val)
            ModifyPointLoads(self.primal_model_part, x_val)
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

        gradient_models = {}
        for node in self.primal_model_part.Nodes:
            node_gradients = np.array([sample_gradient[i][node.Id] for i in range(self.num_samples)])
            node_gradients = node_gradients.reshape(self.num_samples, -1)
            pce_model_gradient = cp.fit_regression(poly_expansion, samples_array, node_gradients)
            gradient_models[node.Id] = pce_model_gradient

        robust_gradient = {}
        for node_id, pce_model_gradient in gradient_models.items():
            mean_gradient = cp.E(pce_model_gradient, distribution)

            # Calculate the gradient of the standard deviation
            std_dev_gradient = np.zeros_like(mean_gradient)
            pce_coefficients = pce_model_gradient.coefficients
            if std_dev > 0:
                for r, coeff in enumerate(pce_coefficients):
                    for k in range(self.num_samples):
                        std_dev_gradient += (coeff ** 2) * (samples_array[r, k] ** 2) * sample_gradient[k][node_id]
                std_dev_gradient /= std_dev

            robust_gradient[node_id] = self.mean_weight * mean_gradient + self.std_weight * std_dev_gradient

        for node in self.primal_model_part.Nodes:
            node.SetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY, robust_gradient[node.Id])

        self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = value
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for calculating the response value", round(timer.time() - startTime, 2), "s")


    def CalculateGradient(self):
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Starting gradient calculation for response", self.identifier)
        startTime = timer.time()
        Logger.PrintInfo("RobustPCEStrainEnergyResponseFunction", "Time needed for calculating gradients", round(timer.time() - startTime, 2), "s")

