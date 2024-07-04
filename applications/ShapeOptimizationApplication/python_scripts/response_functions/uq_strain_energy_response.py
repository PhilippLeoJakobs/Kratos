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

def _GetModelPart(model, solver_settings):
    model_part_name = solver_settings["model_part_name"].GetString()
    if not model.HasModelPart(model_part_name):
        model_part = model.CreateModelPart(model_part_name, 2)
        domain_size = solver_settings["domain_size"].GetInt()
        if domain_size < 0:
            raise Exception('Please specify a "domain_size" >= 0!')
        model_part.ProcessInfo.SetValue(KratosMultiphysics.DOMAIN_SIZE, domain_size)
    else:
        model_part = model.GetModelPart(model_part_name)
    return model_part

def ModifyPointLoads(mp, new_load_x):
    smp = mp.GetSubModelPart("PointLoad3D_load")
    for node in smp.Nodes:
        node.SetSolutionStepValue(StructuralMechanicsApplication.POINT_LOAD,0,new_load_x)

class UQStrainEnergyResponseFunction(ResponseFunctionInterface):
    def __init__(self, identifier, response_settings, model):
        self.identifier = identifier
        self.response_settings = response_settings
        with open(response_settings["primal_settings"].GetString()) as parameters_file:
            ProjectParametersPrimal = Parameters(parameters_file.read())
        self.ProjectParametersPrimal = ProjectParametersPrimal
        self.primal_model_part = _GetModelPart(model, ProjectParametersPrimal["solver_settings"])
        self.model = model
        self.distribution_parameters = response_settings["distribution_parameters"]
        self.primal_analysis = StructuralMechanicsAnalysis(self.model, ProjectParametersPrimal)
        self.primal_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.SHAPE_SENSITIVITY)
        self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, response_settings)

    def Initialize(self):
        self.primal_analysis.Initialize()
        self.response_function_utility.Initialize()

    def InitializeSolutionStep(self):
        self.primal_analysis.time = self.primal_analysis._GetSolver().AdvanceInTime(self.primal_analysis.time)
        self.primal_analysis.InitializeSolutionStep()

    def CalculateValue(self):
        raise NotImplementedError("CalculateValue must be implemented in the derived class")

    def CalculateGradient(self):
        raise NotImplementedError("CalculateGradient must be implemented in the derived class")

    def FinalizeSolutionStep(self):
        self.primal_analysis.FinalizeSolutionStep()
        self.primal_analysis.OutputSolutionStep()

    def Finalize(self):
        self.primal_analysis.time = self.primal_analysis._GetSolver().AdvanceInTime(self.primal_analysis.time)
        self.primal_analysis.InitializeSolutionStep()
        extra_samples =  self.extra_samples
        distribution = generate_distribution(self.distribution_parameters)
        additional_sample_angles = generate_samples(distribution, extra_samples, "latin_hypercube")
        additional_sample_force = calculate_force_vectors(additional_sample_angles, magnitude=100000)
        additional_values = np.zeros(extra_samples)

        for i in range(extra_samples):
            x_val = additional_sample_force[i]
            ModifyPointLoads(self.primal_model_part, x_val)
            self.primal_analysis._GetSolver().Predict()
            self.primal_analysis._GetSolver().SolveSolutionStep()
            self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, self.response_settings)
            self.response_function_utility.Initialize()
            additional_values[i] = self.response_function_utility.CalculateValue()

        self.SaveToCSV(additional_values)
        self.GenerateViolinPlot(additional_values)
        self.primal_analysis.Finalize()

    def SaveToCSV(self, data, filename="additional_values.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Sample Index", "Objective Value"])
            for i, value in enumerate(data):
                writer.writerow([i, value])
        Logger.PrintInfo(self.identifier, f"Additional values saved to {filename}")

    def GenerateViolinPlot(self, data):
        sns.violinplot(data=data)
        plt.title('Violin Plot of Objective Function')
        plt.xlabel('Sample Index')
        plt.ylabel('Objective Value')
        plt.show()

    def GetValue(self):
        return self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE]

    def GetNodalGradient(self, variable):
        if variable != KratosMultiphysics.SHAPE_SENSITIVITY:
            raise RuntimeError(f"GetNodalGradient: No gradient for {variable.Name}!")
        gradient = {node.Id: node.GetSolutionStepValue(variable) for node in self.primal_model_part.Nodes}
        return gradient

    def GetElementalGradient(self, variable):
        raise NotImplementedError("GetElementalGradient needs to be implemented for StrainEnergyResponseFunction")
