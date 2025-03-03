import os

os.environ["WANDB_MODE"] = "offline"

# Env setup
GBA_ROM = "./GBA_ROM/Rockman Zero 3 (Japan).gba"
GBA_SAV = "./GBA_ROM/Rockman Zero 3 (Japan).sav"
RENDER_MODE = "rgb_array"
FRAMESKIP = 1
MAX_TIME = 60 * 60 * 60 # 1 hours of frames
SILENCE = True
FRAMESTACK = 3
RESIZE = True
GRAYSCALE = True

# NN setup
HIDDEN_SIZE = 256
NUM_LAYERS = 1

# ES setup
NUM_ACTORS = os.cpu_count()
POPSIZE = NUM_ACTORS * 4
GENERATIONS = 750

# PPO setup
TOTAL_TIMESTEPS = (MAX_TIME // (1 + FRAMESKIP)) * POPSIZE
N_STEPS = (MAX_TIME // (1 + FRAMESKIP)) // GENERATIONS
BATCH_SIZE = N_STEPS

