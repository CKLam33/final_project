import wandb
wandb.login(host="http://localhost:8080")

# Env setup
GBA_ROM = "./GBA_ROM/Rockman Zero 3 (Japan).gba"
GBA_SAV = "./GBA_ROM/Rockman Zero 3 (Japan).sav"
SILENCE = True
MAX_TIME = 60 * 60 # 1 hours of game play
FRAMESKIP = (3, 5)
OBS_NORM = True
INCLUDE_LIVES = False
RESIZE = True
SCRN_W = 64
SCRN_H = 64
GARYSCALE = True

# LSTM setup
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# PGPE setup
NUM_EPD = 1
NUM_ACTORS = 200
POPSIZE = NUM_ACTORS
GENERATIONS = 500

# Clipup setup
# Follow setup from
# Parameter-exploring Policy GradientsClipUp: A Simple and Powerful Optimizer for
# Distribution-based Policy Evolution
# by Toklu et al. (2020)
STD_LR = 0.1
MAX_SPD = 0.3 # max speed for Clipup
CENTER_LR = MAX_SPD / 2
MOMENTUM = 0.9
STD_MAX_CHG = 0.3

#if number_interaction is used in PGPE
POP_MAX = POPSIZE * 8
NUM_INTERACT = POPSIZE * MAX_TIME * (60 / 4) * 0.75


# Recurrent PPO setup
TOTAL_TIMESTEPS = POPSIZE * MAX_TIME * (60 / 4)
