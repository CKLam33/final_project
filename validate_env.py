import mgba.log
from stable_baselines3.common.env_checker import check_env
from train import load_game
# from pygba import PyGBAEnv
from RMZ3_wrapper import RockmanZero3Wrapper
from make_env import make_game_env

import mgba
mgba.log.silence()

game = load_game('./GBA_ROM/Rockman Zero 3 (Japan).gba', './GBA_ROM/Rockman Zero 3 (Japan).sav')
env = make_game_env(game, RockmanZero3Wrapper(), use_WarpFrame = True, frameskip = 5, seed = 0)
check_env(env)
