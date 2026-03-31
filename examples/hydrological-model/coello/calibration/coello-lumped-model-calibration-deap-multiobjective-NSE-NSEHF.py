# Lumped Model Calibration

# - Please change the Path in the following cell to the directory where you stored the case study data

### Modules
import datetime as dt

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import random

import statista.descriptors as metrics
from deap import algorithms, base, creator, tools

from Hapi.calibration import Calibration
from Hapi.routing import Routing
from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from Hapi.run import Run

# %% Paths
# Parameterpath = path + "examples/hydrological-model/data/lumped_model/Coello_Lumped2021-03-08_muskingum.txt"
MeteoDataPath = "examples/hydrological-model/data/lumped_model/meteo_data-MSWEP.csv"
Path = "examples/hydrological-model/data/lumped_model/"
# %% Meteorological data

start = "2009-01-01"
end = "2011-12-31"
name = "Coello"

Coello = Calibration(name, start, end)
Coello.read_lumped_inputs(MeteoDataPath)

### Basic_inputs

# catchment area
AreaCoeff = 1530
# temporal resolution
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]
# no snow subroutine
Snow = False
Coello.read_lumped_model(HBVLumped, AreaCoeff, InitialCond)

# Calibration parameters

# Calibration boundaries
UB = pd.read_csv(Path + "/UB-3.txt", index_col=0, header=None)
parnames = UB.index
UB = UB[1].tolist()
LB = pd.read_csv(Path + "/LB-3.txt", index_col=0, header=None)
LB = LB[1].tolist()

Maxbas = True
Coello.read_parameters_bound(UB, LB, Snow, maxbas=Maxbas)

### Additional arguments

parameters = []
# Routing
Route = 1
RoutingFn = Routing.triangular_routing_1

# outlet discharge
Coello.read_discharge_gauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
# %% Calibration
creator.create("Fitness", base.Fitness, weights=(1.0, 1.0))
creator.create("IndividualContainer", list, fitness=creator.Fitness)
toolbox = base.Toolbox()


def initializer():
    bounds = []
    for i in range(len(UB)):
        bounds.append(random.uniform(LB[i], UB[i]))
    return bounds


toolbox.register("initialRange", initializer)
toolbox.register(
    "individual", tools.initIterate, creator.IndividualContainer, toolbox.initialRange
)
print(toolbox.individual())
# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
print(toolbox.population(1))

Snow = 0  # no snow subroutine

Coello.OFArgs = []


def objfn(individual):
    # Coello.readParameters(Parameterpath, Snow)
    Coello.Parameters = individual
    Run.runLumped(Coello, Route, RoutingFn)
    # [Coello.QGauges.columns[-1]]
    NSE = metrics.nse_hf(Coello.QGauges, Coello.Qsim, *Coello.OFArgs)
    NSEHF = metrics.nse_hf(Coello.QGauges, Coello.Qsim, *Coello.OFArgs)
    return NSE, NSEHF


def feasible(individual):
    # feasibility function for the indivdual. returns True if feasible, False if otherwise
    if any(individual < np.array(LB)) or any(individual > np.array(UB)):
        # print(indiv)
        return False
    return True


def distance(individual):
    # comes here first and then feasible is called
    # A distance function to the feasibility region
    dist = 0.0
    for i in range(len(individual)):
        penalty = 0
        if individual[i] < LB[i]:
            penalty = LB[i] - individual[i]
        if individual[i] > UB[i]:
            penalty = individual[i] - UB[i]
        dist = dist + penalty

    return dist


toolbox.register("evaluate", objfn)
toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 0, distance))
# try a custom one
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(64)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
CXPB, MUTPB, NGEN = 0.7, 0.2, 250
pop = toolbox.population(n=100)
# %%
algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 140, halloffame=hof, verbose=1)
best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
# %% Run the Model

Coello.Parameters = best_ind
# [0.7686518278956287, 144.35510831203874, 1.9922719933560913, 0.1439126168555068, 0.9474744708723734,
#                  0.749219030317463, 0.8074091462437563, 0.07289588281400794, 68.83482640397304, 5.123384184968337,
#                  1.9922719933560913]
Run.runLumped(Coello, Route, RoutingFn)

### Calculate Performance Criteria

Metrics = dict()

Qobs = Coello.QGauges[Coello.QGauges.columns[0]]

Metrics["RMSE"] = metrics.rmse(Qobs, Coello.Qsim["q"])
Metrics["NSE"] = metrics.nse(Qobs, Coello.Qsim["q"])
Metrics["NSEhf"] = metrics.nse_hf(Qobs, Coello.Qsim["q"])
Metrics["KGE"] = metrics.kge(Qobs, Coello.Qsim["q"])
Metrics["WB"] = metrics.wb(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(Metrics["RMSE"], 2)))
print("NSE= " + str(round(Metrics["NSE"], 2)))
print("NSEhf= " + str(round(Metrics["NSEhf"], 2)))
print("KGE= " + str(round(Metrics["KGE"], 2)))
print("WB= " + str(round(Metrics["WB"], 2)))
# %% Plotting Hydrograph

gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
Coello.plot_hydrograph(plotstart, plotend, gaugei, title="Lumped Model")

# %% Save the Parameters

ParPath = (
    Path
    + f"{Coello.name}-lumped-parameters-multi-obj"
    + str(dt.datetime.now())[0:10]
    + ".txt"
)
parameters = pd.DataFrame(index=parnames)
# parameters['values'] = cal_parameters[1]
# parameters.to_csv(ParPath, header=None, float_format="%0.4f")

# %% Save Results

StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = (
    Path
    + f"{Coello.name}-results-lumped-model-multi-obj"
    + str(dt.datetime.now())[0:10]
    + ".txt"
)
Coello.save_results(result=5, start=StartDate, end=EndDate, path=Path)
