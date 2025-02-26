from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, Union

from env_setup import *
import mlflow
import numpy as np
from RMZ3Env import make_RMZ3Env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3 import PPO
import torch
import torch.nn as nn

torch.set_num_threads(NUM_ACTORS)

PATH = Path("RL/PPO")
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
            max_episode_steps = MAX_STEPS,
            mgba_silence = SILENCE,
            use_framestack = 0,
            to_resize = RESIZE,
            to_grayscale = True
            )
        return env
    return _init

envs = [make_env() for _ in range(POPSIZE)]

envs = VecFrameStack(SubprocVecEnv(envs, start_method="fork"), FRAMESTACK) # use framestack from SB3 instead

policy_kwargs = {
    "activation_fn": nn.ReLU,
}

checkpoint_callback = CheckpointCallback(
  save_freq = 500,
  save_path = PATH.joinpath("/checkpoints/"),
  name_prefix = "ppo_model",
)

# Setup agent and train
agent = PPO("CnnPolicy",
                    envs,
                    verbose = 1,
                    policy_kwargs = policy_kwargs,
                    device = "cuda:0"
                    )

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
