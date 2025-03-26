from datetime import datetime
from pathlib import Path
import lzma, pickle
import numpy as np

import torch
from CNNLSTM import CNNLSTM
from evotorch.algorithms import PGPE
from evotorch.neuroevolution import GymNE
from evotorch.logging import WandbLogger, StdOutLogger

from env_setup import *
from RMZ3Env import make_RMZ3Env

torch.set_num_threads(NUM_ACTORS)

PATH = Path("EA/PGPE_CNNLSTM_distributed")
PATH.mkdir(parents=True, exist_ok=True)

cfg = {
    "env_config":{
        "gba_rom": GBA_ROM,
        "gba_sav": GBA_SAV,
        "max_run_time": MAX_TIME,
        "include_lives_count": INCLUDE_LIVES,
        "render_mode": "rgb_array",
        "frameskip": FRAMESKIP,
        "mgba_silence": SILENCE,
        "to_resize": RESIZE,
        "scrn_w": SCRN_W,
        "scrn_h": SCRN_H,
        "to_grayscale": GARYSCALE,
        "record": False,
        "record_path": PATH.joinpath("videos/")},
    "network": CNNLSTM,
    "network_args": {
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        },
    "obs_norm": OBS_NORM,
    "num_episodes": NUM_EPD,
    "num_actors": NUM_ACTORS,
    "popsize": POPSIZE,
    "center_learning_rate": CENTER_LR,
    "stdev_learning_rate": STD_LR,
    "optimizer": "clipup",
    "optimizer_config": {
        "max_speed": MAX_SPD,
        "stepsize": CENTER_LR,
        "momentum": MOMENTUM,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
    "distributed": True,
    "popsize_weighted_grad_avg": True,
    "stdev_max_change": STD_MAX_CHG,
    "generations": GENERATIONS
    }

problem = GymNE(
    env = make_RMZ3Env,
    env_config = cfg["env_config"],
    network = cfg["network"],
    network_args = cfg["network_args"],
    observation_normalization = cfg["obs_norm"],
    num_episodes = cfg["num_episodes"],
    num_actors = cfg["num_actors"],
)

# Get number of trainable parameters
const = problem.network_constants()
model = CNNLSTM(obs_shape=const['obs_shape'],
                act_length=const["act_length"],
                hidden_size=const["hidden_size"],
                num_layers=const['num_layers'])
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
del model

# Define radius for setup stdev_init
cfg["center_init"] = np.zeros(trainable_params, dtype='float32')
radius = cfg["optimizer_config"]["max_speed"] * 15
cfg["stdev_init"] = np.sqrt((radius ** 2) / trainable_params)

# Initialize PGPE
searcher = PGPE(
    problem,
    popsize = cfg["popsize"],
    stdev_init = cfg["stdev_init"],
    center_learning_rate = cfg["center_learning_rate"],
    stdev_learning_rate = cfg["stdev_learning_rate"],
    optimizer = cfg["optimizer"],
    optimizer_config = cfg["optimizer_config"],
    center_init = cfg["center_init"],
    distributed = cfg["distributed"],
    popsize_weighted_grad_avg = cfg["popsize_weighted_grad_avg"],
    stdev_max_change = cfg["stdev_max_change"],
)

# Add loggers
StdOutLogger(searcher)  # Report the evolution's progress to standard output

# wandb
WandbLogger(searcher,
            id = "PGPE",
            project = "PGPE_CNNLSTM_distributed",
            sync_tensorboard=True,
            monitor_gym=True)

# Run the evolution with record training time
start_time = datetime.now()
searcher.run(cfg["generations"])
end_time = datetime.now()

# Save solution from searcher
with lzma.open(PATH.joinpath(f"searcher_status_{datetime.now()}.xz"), "wb", preset = lzma.PRESET_EXTREME) as f:
    pickle.dump(searcher.status, f)

with open(PATH.joinpath(f"training_duration_{datetime.now()}.txt"), "w", encoding="utf-8") as f:
    f.write(f"Time taken in training:{end_time - start_time}")
    f.close()
