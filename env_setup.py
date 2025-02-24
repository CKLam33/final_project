import os

# Env setup
GBA_ROM = "./GBA_ROM/Rockman Zero 3 (Japan).gba"
GBA_SAV = "./GBA_ROM/Rockman Zero 3 (Japan).sav"
RENDER_MODE = "rgb_array"
FRAMESKIP = 3
MAX_STEPS = 60 * 60 * 60 // (FRAMESKIP + 1) # 3 hours of frames / (frameskip + 1 action)
SILENCE = True
RESIZE = True
GRAYSCALE = True

# NN setup
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# ES setup
NUM_ACTORS = 25
POPSIZE = 50
GENERATIONS = 1000

# PPO setup
TOTAL_TIMESTEPS = MAX_STEPS * GENERATIONS

