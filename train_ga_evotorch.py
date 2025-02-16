from CNNLSTM import CNNLSTM
from env_setup import *
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import Cosyne
from evotorch.logging import StdOutLogger, PandasLogger, MlflowLogger
import gymnasium as gym
import mlflow
from RMZ3Env import make_RMZ3Env

problem = GymNE(
    env=make_RMZ3Env,
    env_config={
        "gba_rom": GBA_ROM,
        "gba_sav": GBA_SAV,
        "render_mode": RENDER_MODE,
        "obs_type": OBS_TYPE,
        "frameskip": FRAMESKIP,
        "mgba_silence": SILENCE,
        "resize": RESIZE},
    network=CNNLSTM,
    network_args={
        "hidden_size": 128,
        "num_layers": 3
    },
    num_episodes = 3,
    num_actors=16,
)

population_size = 100

searcher = Cosyne(
    problem,
    popsize = population_size,
    tournament_size = population_size // 5,
    mutation_stdev = 1,
    mutation_probability = 0.05,
    permute_all = False,
    elitism_ratio = 0.2,
    eta = 5
)

StdOutLogger(searcher)  # Report the evolution's progress to standard output
pandas_logger = PandasLogger(searcher)  # Log the evolution's progress to a pandas DataFrame
mlflow.set_registry_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()
run = mlflow.start_run()
MlflowLogger(searcher, client=client, run=run)

searcher.run(500)
solutions = searcher.status

df = pandas_logger.to_dataframe()
df.to_csv("GA_results.csv", index=False)
