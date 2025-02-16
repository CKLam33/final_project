import sys
from typing import Any, Dict, Tuple, Union

from env_setup import *
from gymnasium.wrappers import ResizeObservation
import mlflow
import numpy as np
from RMZ3Env import make_RMZ3Env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from sb3_contrib import RecurrentPPO, MaskablePPO


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

loggers = Logger(
    folder="./RL-model/",
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

# Create environments
def make_env():
    def _init():
        env = make_RMZ3Env(
            gba_rom = GBA_ROM,
            gba_sav = GBA_SAV,
            render_mode = RENDER_MODE,
            obs_type = OBS_TYPE,
            frameskip = FRAMESKIP,
            mgba_silence = SILENCE,
            resize = RESIZE,
            )
        return env
    return _init

envs = [make_env() for _ in range(16)]

envs = SubprocVecEnv(envs, start_method="fork")

policy_kargs = {
    "lstm_hidden_size": 128,
    "n_lstm_layers": 3,
}

n_steps = (100 * 3) // 16 # popluation size * episode per generation / number of actors

with mlflow.start_run():
# Setup agent and train
    agent = RecurrentPPO("CnnLstmPolicy",
                        envs,
                        n_steps = n_steps,
                        learning_rate = 1e-4,
                        ent_coef = 0.001,
                        target_kl = 1e-4,
                        verbose = 2)

    # Time steps per episode = frames to finish one checkpoint / frames (frame skip + 1 rendered) per timestep
    timesteps_per_episode = (2 * 60 * 60) // (FRAMESKIP + 1)
    timesteps = 500 * 3 * timesteps_per_episode
    agent.set_logger(loggers)
    agent.learn(total_timesteps = timesteps, progress_bar=True, log_interval=1)
    agent.save(path="./RecurrentPPO_RMZ3")
    envs.close()
