import unittest

import mgba.log
import gymnasium as gym
import numpy as np
from pygba import PyGBA
from stable_baselines3.common.env_checker import check_env
import torch

from train import load_game
from make_env import make_game_env
from RMZ3_wrapper import RockmanZero3Wrapper

import mgba

mgba.log.silence()

class TestFunctions(unittest.TestCase):
    def test_load_game_without_save(self):
        game = load_game("./GBA_ROM/Rockman Zero 3 (Japan).gba")
        self.assertIsInstance(game, PyGBA)

    def test_load_game_with_save(self):
        game = load_game("./GBA_ROM/Rockman Zero 3 (Japan).gba", "./GBA_ROM/Rockman Zero 3 (Japan).sav")
        self.assertIsInstance(game, PyGBA)

    def test_read_starting_options(self):
        game = load_game("./GBA_ROM/Rockman Zero 3 (Japan).gba", "./GBA_ROM/Rockman Zero 3 (Japan).sav")
        for _ in range(4):
            game.wait(10)
            game.press_start(10)
        game.wait(10)
        second_option_a = game.read_u16(0x02030318)
        second_option_b = game.read_u16(0x0203031C)
        self.assertEqual(second_option_a, 18434)
        self.assertEqual(second_option_b, 18442)
        game.press_up(10)
        game.wait(10)
        first_option_a = game.read_u16(0x02030318)
        first_option_b = game.read_u16(0x0203031C)
        self.assertEqual(first_option_a, 18426)
        self.assertEqual(first_option_b, 18450)
        

    def test_make_game_env(self):
        game = load_game('./GBA_ROM/Rockman Zero 3 (Japan).gba', './GBA_ROM/Rockman Zero 3 (Japan).sav')
        env = make_game_env(game, RockmanZero3Wrapper(), use_WarpFrame = True)
        self.assertIsInstance(env, gym.Env)
        self.assertEqual(env.observation_space.shape, (80, 120, 1))
        self.assertIsNone(check_env(env))

    def test_env_actions(self):
        game = load_game('./GBA_ROM/Rockman Zero 3 (Japan).gba', './GBA_ROM/Rockman Zero 3 (Japan).sav')
        env = make_game_env(game, RockmanZero3Wrapper(), use_WarpFrame = True)
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        self.assertEqual(env.action_space.n, 35)

    def test_gpu_available(self):
        self.assertTrue(torch.cuda.is_available())

if __name__ == '__main__':
    unittest.main()
    