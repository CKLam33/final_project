import os.path
import numpy as np
import mgba
from pygba import PyGBA
from make_env import make_game_env
from RMZ3_wrapper import RockmanZero3Wrapper
from sb3_contrib import RecurrentPPO

mgba.log.silence()

def load_game(file: str, save_file:str | None = None) -> PyGBA:
    assert file.endswith(".gba")
    assert os.path.exists(file)

    if save_file is not None:
        assert save_file.endswith(".sav")
        assert os.path.exists(save_file)
    else:
        save_file = None

    return PyGBA.load(file, save_file = save_file)

def main():
    game = load_game("./GBA_ROM/Rockman Zero 3 (Japan).gba", "./GBA_ROM/Rockman Zero 3 (Japan).sav")
    for _ in range(4):
        game.wait(10)
        game.press_start(10)
    game.wait(10)
    if game.read_u16(0x02030318) == 18434 and game.read_u16(0x0203031C) == 18442:
        game.press_up(10)
        game.wait(10)
    if game.read_u16(0x02030318) == 18426 and game.read_u16(0x0203031C) == 18450:
        game.press_a(10)
        game.wait(10)
    for _ in range(2):
        game.press_start(10)
        game.wait(10)
    env = make_game_env(game, RockmanZero3Wrapper(), use_WarpFrame = True)
    model = RecurrentPPO("CnnLstmPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    


if __name__ == "__main__":
    main()