import KratosMultiphysics
from KratosMultiphysics import Parameters, Logger
from KratosMultiphysics.response_functions.response_function_interface import ResponseFunctionInterface
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

import numpy as np
import time as timer

def _GetModelPart(model, solver_settings):
    #TODO can be removed once model is fully available
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


def ModifyPointLoads(mp, new_load_x, new_load_y, new_load_z):
    for node in mp.Nodes:
        node.SetSolutionStepValue(StructuralMechanicsApplication.POINT_LOAD,0,[new_load_x,new_load_y,new_load_z])



# ==============================================================================
class MeanMCStrainEnergyResponseFunction(ResponseFunctionInterface):
    """Linear strain energy response function. It triggers the primal analysis and
    uses the primal analysis results to calculate response value and gradient.

    Attributes
    ----------
    primal_model_part : Model part of the primal analysis object
    primal_analysis : Primal analysis object of the response function
    response_function_utility: Cpp utilities object doing the actual computation of response value and gradient.
    """

    def __init__(self, identifier, response_settings, model):
        self.identifier = identifier
        self.response_settings=response_settings
        with open(response_settings["primal_settings"].GetString()) as parameters_file:
            ProjectParametersPrimal = Parameters(parameters_file.read())

        self.primal_model_part = _GetModelPart(model, ProjectParametersPrimal["solver_settings"])

        self.primal_analysis = StructuralMechanicsAnalysis(model, ProjectParametersPrimal)
        self.primal_model_part.AddNodalSolutionStepVariable(KratosMultiphysics.SHAPE_SENSITIVITY)

        self.num_samples = 100

        self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, response_settings)

    def Initialize(self):
        self.primal_analysis.Initialize()
        self.response_function_utility.Initialize()

    def InitializeSolutionStep(self):

        self.primal_analysis.time = self.primal_analysis._GetSolver().AdvanceInTime(self.primal_analysis.time)
        self.primal_analysis.InitializeSolutionStep()

    def CalculateValue(self):
        Logger.PrintInfo("StrainEnergyResponse", "Starting primal analysis for response", self.identifier)
       

        startTime = timer.time()
        self.primal_analysis._GetSolver().Predict()
        self.primal_analysis._GetSolver().SolveSolutionStep()
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for solving the primal analysis",round(timer.time() - startTime,2),"s")

        startTime = timer.time()
        sample_value={}
        for i in range(self.num_samples):
            y_val = np.random.normal(-100000,10000)
            x_val = np.random.normal(0,60000)
            ModifyPointLoads(self.primal_model_part,x_val,y_val,0)
            self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, self.response_settings)
            sample_value[i] = self.response_function_utility.CalculateValue()
        value = np.mean(sample_value)

        self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = value
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating the response value",round(timer.time() - startTime,2),"s")

    def CalculateGradient(self):
        Logger.PrintInfo("StrainEnergyResponse", "Starting gradient calculation for response", self.identifier)

        startTime = timer.time()
        sample_gradient={}
        for i in range(self.num_samples):
            y_val = np.random.normal(-100000,10000)
            x_val = np.random.normal(0,60000)
            ModifyPointLoads(self.primal_model_part,x_val,y_val,0)
            self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, self.response_settings)
            sample_gradient[i] = self.response_function_utility.CalculateGradient()
        gradient = np.mean(sample_gradient)
        self.response_function_utility.CalculateGradient()
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating gradients",round(timer.time() - startTime,2),"s")

    def FinalizeSolutionStep(self):
        self.primal_analysis.FinalizeSolutionStep()
        self.primal_analysis.OutputSolutionStep()

    def Finalize(self):
        self.primal_analysis.Finalize()

    def GetValue(self):
        return self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE]

    def GetNodalGradient(self, variable):
        if variable != KratosMultiphysics.SHAPE_SENSITIVITY:
            raise RuntimeError("GetNodalGradient: No gradient for {}!".format(variable.Name))
        gradient = {}
        for node in self.primal_model_part.Nodes:
            gradient[node.Id] = node.GetSolutionStepValue(variable)
        return gradient

    def GetElementalGradient(self, variable):
        raise NotImplementedError("GetElementalGradient needs to be implemented for StrainEnergyResponseFunction")