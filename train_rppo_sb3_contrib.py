from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, Union

from env_setup import *
import mlflow
import numpy as np
from RMZ3Env import make_RMZ3Env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from sb3_contrib import RecurrentPPO
import torch
import torch.nn as nn

torch.set_num_threads(NUM_ACTORS)

PATH = Path("RL/RecurrentPPO")
PATH.mkdir(parents=True, exist_ok=True)

class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)

logger = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

# Create environments
def make_env():
    def _init():
        env = make_RMZ3Env(
            gba_rom = GBA_ROM,
            gba_sav = GBA_SAV,
            render_mode = RENDER_MODE,
            frameskip = FRAMESKIP,
            max_episode_steps = MAX_STEPS,
            mgba_silence = SILENCE,
            to_resize = RESIZE,
            to_grayscale = True
            )
        return env
    return _init

envs = [make_env() for _ in range(POPSIZE)]

envs = SubprocVecEnv(envs, start_method="fork")

policy_kwargs = {
    "lstm_hidden_size": HIDDEN_SIZE,
    "n_lstm_layers": NUM_LAYERS,
    "activation_fn": nn.ReLU,
}

checkpoint_callback = CheckpointCallback(
  save_freq = 1080,
  save_path = PATH.joinpath("/checkpoints/"),
  name_prefix = "rppo_model",
)

# Setup agent and train
agent = RecurrentPPO("CnnLstmPolicy",
                    envs,
                    verbose = 1,
                    policy_kwargs = policy_kwargs,
                    )

timesteps = MAX_STEPS * GENERATIONS # (Average gameplay duration over Frameskip) * generations
agent.set_logger(logger)
start_time = datetime.now()
agent.learn(total_timesteps = TOTAL_TIMESTEPS,
            progress_bar=True,
            log_interval=10,
            callback=checkpoint_callback)

agent.save(path = PATH.joinpath("/final_model/"))
    
end_time = datetime.now()
f = open(PATH.joinpath("training_duration.txt"), "w+")
f.write(f"Time taken in training:{end_time - start_time}")
f.close
envs.close()
