from datetime import datetime
from pathlib import Path
import pickle

from CNNLSTM import CNNLSTM
from env_setup import *
from evotorch.algorithms import Cosyne
from evotorch.neuroevolution import GymNE
from evotorch.logging import MlflowLogger, PicklingLogger, StdOutLogger
import mlflow
from RMZ3Env import make_RMZ3Env
import torch

torch.set_num_threads(NUM_ACTORS)

PATH = Path("EA/Cosyne_CNN")
PATH.mkdir(parents=True, exist_ok=True)

problem = GymNE(
    env=make_RMZ3Env,
    env_config={
        "gba_rom": GBA_ROM,
        "gba_sav": GBA_SAV,
        "render_mode": RENDER_MODE,
        "max_episode_steps": MAX_STEPS,
        "mgba_silence": SILENCE,
        "resize": RESIZE,
        "to_grayscale": False,
        "use_framestack": 4,
        "record": False,
        "record_path": PATH.joinpath("videos/")},
    network=CNNLSTM,
    network_args={
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "is_bidirectional": False,
    },
    episode_length = MAX_STEPS,
    num_actors = NUM_ACTORS,
)

# Initialize PGPE
searcher = Cosyne(
    problem,
    popsize = POPSIZE,
    tournament_size = 4,
    mutation_stdev = 0.3,
    mutation_probability = 0.5,
    num_elites = 1
)

# Add loggers
StdOutLogger(searcher)  # Report the evolution's progress to standard output

# mlflow
client = mlflow.tracking.MlflowClient()
run = mlflow.start_run(run_name = "Cosyne_CNNLSTM_RMZ3")
MlflowLogger(searcher, client = client, run=run)
PicklingLogger(searcher, interval = 10,
               directory = PATH.joinpath("pop_best_sol"),
               items_to_save = "pop_best_eval")

# Run the evolution with record training time
start_time = datetime.now()
searcher.run(GENERATIONS)
end_time = datetime.now()

PATH.mkdir(parents=True, exist_ok=True)
with open(PATH.joinpath("training_duration.txt"), "w") as f:
    f.write(f"Time taken in training:{end_time - start_time}")
    f.close()
