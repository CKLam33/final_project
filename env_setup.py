import os

# Env setup
GBA_ROM = "./GBA_ROM/Rockman Zero 3 (Japan).gba"
GBA_SAV = "./GBA_ROM/Rockman Zero 3 (Japan).sav"
RENDER_MODE = "rgb_array"
FRAMESKIP = 0
MAX_STEPS = 60 * 60 * 60 # 1 hours of frames
SILENCE = True
FRAMESTACK = 4
RESIZE = True
GRAYSCALE = True

# NN setup
HIDDEN_SIZE = 256
NUM_LAYERS = 1

# ES setup
NUM_ACTORS = 25
POPSIZE = NUM_ACTORS * 2
GENERATIONS = 2500

# PPO setup
TOTAL_TIMESTEPS = MAX_STEPS * GENERATIONS

