
from gymnasium.wrappers import ResizeObservation

from RMZ3Env import make_RMZ3Env
from mgba.log import silence

from CNNLSTM_EvoTorch import CNNLSTM
from evotorch.neuroevolution import GymNE
from evotorch.algorithms import Cosyne
from evotorch.logging import StdOutLogger, PandasLogger, MlflowLogger
import gymnasium as gym
import mlflow

silence()

# Create the environment
def make_env(env_name: str,
             gba_rom: str,
             gba_sav: str,
             obs_type: str,
             render_mode: str,
             frameskip: int,
             mgba_silence: bool = False,
             resize: bool = True):

    env = gym.make(env_name,
                   gba_rom=gba_rom,
                   gba_sav=gba_sav,
                   obs_type=obs_type,
                   frameskip=frameskip,
                   mgba_silence=mgba_silence,
                   render_mode=render_mode)
    
    if resize:
        env = ResizeObservation(env, (120, 80))
    return env

if __name__ == "__main__":
    problem = GymNE(
        env=make_RMZ3Env,
        env_config={
            "gba_rom": "./GBA_ROM/Rockman Zero 3 (Japan).gba",
            "gba_sav": "./GBA_ROM/Rockman Zero 3 (Japan).sav",
            "render_mode": "rgb_array",
            "obs_type": "grayscale",
            "frameskip": 3,
            "mgba_silence": False,
            "resize": True},
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
    client = mlflow.tracking.MlflowClient()
    run = mlflow.start_run()
    MlflowLogger(searcher, client=client, run=run)

    searcher.run(500)
    solutions = searcher.status

    df = pandas_logger.to_dataframe()
    df.to_csv("GA_results.csv", index=False)