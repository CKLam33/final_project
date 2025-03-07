from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Tuple, Union

from env_setup import *
import numpy as np
from RMZ3Env import make_RMZ3Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3 import PPO
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback

# wandb.login(host="http://localhost:8080")

PATH = Path("RL/PPO")
PATH.mkdir(parents=True, exist_ok=True)

cfg = {
    "env_config":{
        "gba_rom": GBA_ROM,
        "gba_sav": GBA_SAV,
        "max_run_time": MAX_TIME,
        "mgba_silence": SILENCE,
        "use_framestack": FRAMESTACK,
        "record": False,
        "record_path": PATH.joinpath("videos/")},
    "policy": "CnnPolicy",
    "total_timesteps": TOTAL_TIMESTEPS,
    "policy_kwargs": {
            "activation_fn": nn.ReLU,
        },
    "n_steps": N_STEPS,
    "batch_size": BATCH_SIZE,
    }

run = wandb.init(
    id = "Basic PPO",
    project = "PPO",
    config = cfg,
    sync_tensorboard=True,
    monitor_gym=True
    )

# Create environments
def make_env():
    def _init():
        env = make_RMZ3Env(
            gba_rom = cfg["env_config"]["gba_rom"],
            gba_sav = cfg["env_config"]["gba_sav"],
            max_run_time = cfg["env_config"]["max_run_time"],
            mgba_silence = cfg["env_config"]["mgba_silence"],
            use_framestack = 0,
            )
        return Monitor(env)
    return _init

envs = [make_env() for _ in range(NUM_ACTORS)]
envs = SubprocVecEnv(envs, start_method="fork")

envs = VecFrameStack(envs, cfg["env_config"]["use_framestack"]) # use framestack from SB3 instead

# Setup agent and train
agent = PPO(cfg["policy"],
            envs,
            verbose = 1,
            policy_kwargs = cfg["policy_kwargs"],
            n_steps = cfg["n_steps"],
            batch_size = cfg["batch_size"],
            tensorboard_log=f"runs/{run.id}"
            )

start_time = datetime.now()
agent.learn(total_timesteps = cfg["total_timesteps"],
            progress_bar = True,
            log_interval = 1,
            callback = WandbCallback(
            gradient_save_freq = 1,
            model_save_path = f"models/{run.id}",
            model_save_freq = 2048 * NUM_ACTORS * 50,
            verbose = 1,
        ),
)
end_time = datetime.now()

run.finish()
    
f = open(PATH.joinpath("training_duration.txt"), "w+")
f.write(f"Time taken in training:{end_time - start_time}")
f.close
envs.close()
