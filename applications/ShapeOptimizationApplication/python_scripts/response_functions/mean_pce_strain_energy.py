import KratosMultiphysics
from KratosMultiphysics import Parameters, Logger
from KratosMultiphysics.response_functions.response_function_interface import ResponseFunctionInterface
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.ShapeOptimizationApplication.utilities.custom_uq_tools import generate_samples, calculate_force_vectors_x, calculate_force_vectors_z, generate_distribution
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import chaospy as cp
import numpy as np
import time as timer
from .uq_strain_energy_response import UQStrainEnergyResponseFunction, ModifyPointLoads


# ==============================================================================
class MeanPCEStrainEnergyResponseFunction(UQStrainEnergyResponseFunction):
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
        self.extra_samples = response_settings["extra_samples"].GetInt()
        self.pce_order = response_settings["pce_order"].GetInt()
        self.force_direction = response_settings["force_direction"].GetString()

    def CalculateValue(self):

        Logger.PrintInfo("StrainEnergyResponse", "Starting primal analysis for response", self.identifier)

        startTime = timer.time()

        Logger.PrintInfo("StrainEnergyResponse", "Time needed for solving the primal analysis", round(timer.time() - startTime, 2), "s")

        startTime = timer.time()

        distribution = generate_distribution(self.distribution_parameters)
        # Generate samples using generate_downward_vector
        
        sample_angles =generate_samples(distribution,self.num_samples,self.sampling_strategy)
                # Choose the appropriate force vector calculation function
        if self.force_direction.lower() == 'x':
            sample_force= calculate_force_vectors_x(sample_angles, magnitude=100000)
        elif self.force_direction.lower() == 'z':
            sample_force = calculate_force_vectors_z(sample_angles, magnitude=100000)
        else:
            raise ValueError(f"Unknown force direction: {self.force_direction}")

        sample_value = np.zeros(self.num_samples)
        sample_gradient = [{} for _ in range(self.num_samples)]

        for i in range(self.num_samples):
            x_val = sample_force[i]
            Logger.PrintInfo("Sample value: ", x_val)

            # Modify point loads based on random values
            ModifyPointLoads(self.primal_model_part, x_val)

            # Perform structural analysis
            # self.primal_analysis = StructuralMechanicsAnalysis(self.model, self.ProjectParametersPrimal)
            self.primal_analysis._GetSolver().Predict()
            self.primal_analysis._GetSolver().SolveSolutionStep()

            # Initialize and calculate the response function value
            self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, self.response_settings)
            self.response_function_utility.Initialize()
            sample_value[i] = self.response_function_utility.CalculateValue()
            Logger.PrintInfo(sample_value[i])

            # Store the response value in the model part
            self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = sample_value[i]

            # Calculate the gradient
            self.response_function_utility.CalculateGradient()

            # Store the gradient for each node
            for node in self.primal_model_part.Nodes:
                sample_gradient[i][node.Id] = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)

        # Convert samples to a numpy array for PCE fitting
        samples_array = np.array(sample_angles).T

        poly_expansion = cp.orth_ttr(self.pce_order, distribution)
        pce_model_value = cp.fit_regression(poly_expansion, samples_array, sample_value)
        

        # Calculate the mean value of the response using the PCE model
        value = cp.E(pce_model_value, distribution)
        Logger.PrintInfo(value)
        # Create PCE surrogate models for gradients
        gradient_models = {}
        for node in self.primal_model_part.Nodes:
            node_gradients = np.array([sample_gradient[i][node.Id] for i in range(self.num_samples)])
            # Ensure the dimensions are correct for regression fitting
            node_gradients = node_gradients.reshape(self.num_samples, -1)
            pce_model_gradient = cp.fit_regression(poly_expansion, samples_array, node_gradients)
            gradient_models[node.Id] = pce_model_gradient


        # Calculate the mean gradients using the PCE models
        mean_gradient = {}
        for node_id, pce_model_gradient in gradient_models.items():
            mean_gradient[node_id] = cp.E(pce_model_gradient, distribution)


        # Set the gradient to the mean gradient
        for node in self.primal_model_part.Nodes:
            node.SetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY, mean_gradient[node.Id])


        self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = value
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating the response value", round(timer.time() - startTime, 2), "s")


    def CalculateGradient(self):
        Logger.PrintInfo("StrainEnergyResponse", "Starting gradient calculation for response", self.identifier)

        startTime = timer.time()

        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating gradients",round(timer.time() - startTime,2),"s")

 