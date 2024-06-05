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
    smp = mp.GetSubModelPart("PointLoad3D_load")
    for node in smp.Nodes:
        node.SetSolutionStepValue(StructuralMechanicsApplication.POINT_LOAD,0,[new_load_x,new_load_y,new_load_z])



# ==============================================================================
class VarianceMCStrainEnergyResponseFunction(ResponseFunctionInterface):
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
        self.ProjectParametersPrimal=ProjectParametersPrimal
        self.primal_model_part = _GetModelPart(model, ProjectParametersPrimal["solver_settings"])
        self.model=model
        self.primal_analysis = StructuralMechanicsAnalysis(self.model, ProjectParametersPrimal)
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
        
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for solving the primal analysis",round(timer.time() - startTime,2),"s")

        startTime = timer.time()
        
        sample_value = np.zeros(self.num_samples)
        sample_gradient = [{} for _ in range(self.num_samples)]

        for i in range(self.num_samples):
            y_val = np.random.normal(-100000, 10000)
            Logger.PrintInfo(y_val)
            x_val = np.random.normal(0, 60000)
            
            # Modify point loads based on random values
            ModifyPointLoads(self.primal_model_part, x_val, y_val, 0)
            
            # Perform structural analysis
            #self.primal_analysis = StructuralMechanicsAnalysis(self.model, self.ProjectParametersPrimal)
            self.primal_analysis._GetSolver().Predict()
            self.primal_analysis._GetSolver().SolveSolutionStep()
            
            # Initialize and calculate the response function value
            self.response_function_utility = StructuralMechanicsApplication.StrainEnergyResponseFunctionUtility(self.primal_model_part, self.response_settings)
            self.response_function_utility.Initialize()
            sample_value[i] = self.response_function_utility.CalculateValue()
            
            # Store the response value in the model part
            self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = sample_value[i]
            
            # Calculate the gradient
            self.response_function_utility.CalculateGradient()
            
            # Store the gradient for each node
            for node in self.primal_model_part.Nodes:
                sample_gradient[i][node.Id] = node.GetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY)

        mean_gradient = {}
        for grad_dict in sample_gradient:
            for node_id, gradient in grad_dict.items():
                if node_id not in mean_gradient:
                    mean_gradient[node_id] = np.zeros_like(gradient)
                mean_gradient[node_id] += gradient
        for node_id in mean_gradient:
            mean_gradient[node_id] /= self.num_samples

        # Calculate the mean value of the response
        value = np.var(sample_value, dtype=float)

        # Initialize the mean_gradient dictionary
        variance_gradient = {node_id: np.zeros_like(gradient) for node_id in mean_gradient}

        # Sum the gradients from each sample
        for grad_dict in sample_gradient:
            for node_id, gradient in grad_dict.items():
                gradient_np = np.array(gradient)  # Convert Kratos.Array3 to NumPy array
                mean_gradient_np = np.array(mean_gradient[node_id])  # Convert mean gradient to NumPy array
                variance_gradient[node_id] += (gradient_np - mean_gradient_np) ** 2

        # Set the gradient to the mean gradient
        gradient = variance_gradient

        for node in self.primal_model_part.Nodes:
            node.SetSolutionStepValue(KratosMultiphysics.SHAPE_SENSITIVITY,gradient[node.Id])
        

        self.primal_model_part.ProcessInfo[StructuralMechanicsApplication.RESPONSE_VALUE] = value
        Logger.PrintInfo("StrainEnergyResponse", "Time needed for calculating the response value",round(timer.time() - startTime,2),"s")

    def CalculateGradient(self):
        Logger.PrintInfo("StrainEnergyResponse", "Starting gradient calculation for response", self.identifier)

        startTime = timer.time()

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