import os

os.environ["WANDB_MODE"] = "offline"

# Env setup
GBA_ROM = "./GBA_ROM/Rockman Zero 3 (Japan).gba"
GBA_SAV = "./GBA_ROM/Rockman Zero 3 (Japan).sav"
SILENCE = True
MAX_TIME = 60 * 60 * 60 # 1 hours of frames
FRAMESKIP = (0, 1)
FRAMESTACK = 2
OBS_NORM = True
INCLUDE_LIVES = False

# LSTM setup
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# PGPE setup
NUM_ACTORS = 100
POPSIZE = NUM_ACTORS
GENERATIONS = 500
# Follow setup from Parameter-exploring Policy Gradients
# by Sehnke et al. (2010)
STDEV_INIT = 2.0 
CENTER_LR = 0.2
STD_LR = 0.1
# Clipup setup
# Follow setup from
# Parameter-exploring Policy GradientsClipUp: A Simple and Powerful Optimizer for
# Distribution-based Policy Evolution
# by Toklu et al. (2020)
MAX_SPD = CENTER_LR * 2 # max speed for Clipup
STD_MAX_CHG = 0.3
OPT_LR = MAX_SPD / 2

# Recurrent PPO setup
TOTAL_TIMESTEPS = MAX_TIME * POPSIZE
