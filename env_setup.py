import os

os.environ["WANDB_MODE"] = "offline"

# Env setup
GBA_ROM = "./GBA_ROM/Rockman Zero 3 (Japan).gba"
GBA_SAV = "./GBA_ROM/Rockman Zero 3 (Japan).sav"
MAX_TIME = 60 * 60 * 60 # 1 hours of frames
SILENCE = True
FRAMESTACK = 5
OBS_NORM = True
INCLUDE_LIVES = False

# LSTM setup
HIDDEN_SIZE = 256
NUM_LAYERS = 1

# PGPE setup
NUM_ACTORS = POPSIZE = os.cpu_count()
GENERATIONS = 500
RAD_INIT = 5
CENTER_LR = 0.05
STD_LR = 0.1

# PPO setup
TOTAL_TIMESTEPS = MAX_TIME * POPSIZE
N_STEPS = MAX_TIME // GENERATIONS
BATCH_SIZE = 512

