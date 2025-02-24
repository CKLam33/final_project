from itertools import product
import sys
from typing import Any, Literal, Callable, List, Tuple, Dict, Optional
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, RecordVideo
import mgba.image
import numpy as np
from pygba import PyGBA

import pygame

from mgba.gba import GBA
from mgba.log import silence

# GBA control
# Select key is not used in the game
KEY_MAP = {
    "up": GBA.KEY_UP,
    "down": GBA.KEY_DOWN,
    "left": GBA.KEY_LEFT,
    "right": GBA.KEY_RIGHT,
    "A": GBA.KEY_A,
    "B": GBA.KEY_B,
    "L": GBA.KEY_L,
    "R": GBA.KEY_R,
    "start": GBA.KEY_START,
}

# Boss location
# checkpoint: [stages]
BOSS_LOCATION = {
    5: [7, 8, 9, 10, 12],
    6: [1, 4, 6, 11, 13, 14],
    7: [2, 5],
    8: [15],
    9: [3, 16]
}

# Final boss location in last stage
FINAL_STAGE_BOSSES = {
    3: [1, 2, 3, 4],
    6: [5, 6, 7, 8],
    9: [1, 3, 7]
}

# Pillow image to pygame image
def _pil_image_to_pygame(img):
    
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode).convert()

# Basic Rockman Zero 3 Environment
class RMZ3Env(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(
        self,
        gba_rom: str | Path,
        gba_sav: str | Path | None = None,
        mgba_silence: bool = False,
        obs_type: Literal["rgb", "grayscale"] = "rgb",
        frameskip: int | tuple[int, int] | tuple[int, int, int] = 0,
        repeat_action_probability: float = 0.0,
        render_mode: Literal["human", "rgb_array"] | None = "human",
        reset_to_initial_state: bool = True,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        if gba_rom is None:
            raise ValueError("GBA rom is required")
        self.gba = PyGBA.load(gba_rom, gba_sav)
        if mgba_silence:
            silence()
        self.obs_type = obs_type
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Generate all possible product of buttons.
        # So that even the AI change the key setting accidentally,
        # it still about to control the character
        arrow = [None, "up", "down", "right", "left"]
        L = [None, "L"]
        R = [None, "R"]
        A = [None, "A"]
        B = [None, "B"]
        self.actions = list(product(arrow, L, R, A, B))
        self.actions.append((None, None, None, None, "start"))
        self.action_space = Discrete(len(self.actions))

        # Building the observation_space
        screen_size = self.gba.core.desired_video_dimensions()
        if obs_type == "rgb":
            screen_size += (3,)  # RGB channels
        else:  # grayscale
            screen_size += (1,)  # Single channel
        self.observation_space = Box(
            low=0,
            high=255,
            shape=screen_size,
            dtype=np.uint8
        )

        self._framebuffer = mgba.image.Image(*self.gba.core.desired_video_dimensions())
        self.gba.core.set_video_buffer(self._framebuffer)  # need to reset after this

        self._screen = None
        self._clock = None

        self._total_reward = 0
        self._last_total_reward = 0
        self._prev_reward = 0
        self._step = 0

        # Go into first stage
        self.gba.wait(10)
        self.gba.press_start(10)
        self.gba.wait(10)
        self.gba.press_start(10)
        self.gba.wait(10)
        self.gba.press_start(10)
        self.gba.wait(10)
        self.gba.press_start(10)
        self.gba.wait(10)
        if self.gba.read_u16(0x02030318) == 18434 and self.gba.read_u16(0x0203031C) == 18442:
            self.gba.press_up(10)
            self.gba.wait(30)
            if self.gba.read_u16(0x02030318) == 18426 and self.gba.read_u16(0x0203031C) == 18450:
                self.gba.press_a(10)
                self.gba.wait(30)
                self.gba.press_start(10)
                self.gba.wait(30)
                self.gba.press_start(10)
                self.gba.wait(30)

        # Save initial state
        if reset_to_initial_state:
            self._initial_state = self.gba.core.save_raw_state()
            pass
        else:
            self._initial_state = None
            
        self._kwargs = kwargs

        # Initialize state
        self._get_game_data()
        self.total_play_time = 0
        self.same_pos_count = 0
        self.best_stage = self.curr_stage
        self.best_checkpoint = self.curr_checkpoint
        self.prev_stage = self.curr_stage
        self.prev_checkpoint = self.curr_checkpoint

        # Reset the environment
        self.reset()

    def _get_game_data(self):
        self.x_pos = self.gba.read_u16(0x02037CB4)
        self.prev_x_pos = self.gba.read_u16(0x02037D60)
        self.health = self.gba.read_u8(0x02037D04)
        self.life_count = self.gba.read_u8(0x02036F70)
        self.curr_stage = self.gba.read_u8(0x0202FE60)
        self.curr_checkpoint = self.gba.read_u8(0x0202FE62)
        self.checkpoint_time = self.gba.read_u16(0x0202FF40) / 60
        self.exskills_1 = self.gba.read_u16(0x02037D28) & 0x000F
        self.exskills_2 = (self.gba.read_u16(0x02037D28) & 0x00F0) // 16
        self.exskills_3 = (self.gba.read_u16(0x02037D28) & 0x0F00) // 16**2
        self.boss_health = self.gba.read_u16(0x0203BB04)
        self.final_subbosses_room = self.gba.read_u16(0x0202FE6B)

    def get_action_by_id(self, action_id: int) -> tuple[Any, Any]:
        if action_id < 0 or action_id > len(self.actions):
            raise ValueError(f"action_id {action_id} is invalid")
        return self.actions[action_id]

    def get_action_id(self, arrow: str, buttonL: str, buttonR: str, buttonA: str, buttonB: str ) -> int:
        action = (arrow, buttonL, buttonR, buttonA, buttonB)
        if action not in self.actions:
            raise ValueError(f"Invalid action: Must be a tuple of (arrow, button)")
        return self.actions.index(action)

    def _get_observation(self):
        img = self._framebuffer.to_pil()
        if self.obs_type == "grayscale":
            img = img.convert("L")
            return np.expand_dims(np.array(img).transpose(1, 0), -1)
        else:
            img = img.convert("RGB")
            return np.array(img).transpose(1, 0, 2)

    def step(self, action_id):

        self._get_game_data()
        self.total_play_time += ((1 + self.frameskip) / 60)

        actions = self.get_action_by_id(action_id)
        actions = [KEY_MAP[a] for a in actions if a is not None]
        if np.random.random() > self.repeat_action_probability:
            self.gba.core.set_keys(*actions)

        if self.x_pos == self.prev_x_pos:
            self.same_pos_count += 1

        # Skip frames
        if isinstance(self.frameskip, tuple):
            frameskip = np.random.randint(*self.frameskip)
        else:
            frameskip = self.frameskip

        for _ in range(frameskip + 1):
            self.gba.core.run_frame()
        observation = self._get_observation()

        # Calculate reward
        health_life = self.health // 16 + self.life_count // 2 # Reward if still alive
        # Reward start from second checkpoint in first stage and
        # not give any reward if character is in Commander base
        stage_checkpoint = (self.curr_stage // 16 + self.curr_checkpoint // 9) \
                            if (self.curr_stage < 17) and (self.curr_stage >= 1 and self.curr_checkpoint > 3) else 0
        ex_skills = (self.exskills_1 + self.exskills_2 + self.exskills_3) / 48
        boss_killed = 1 if self.check_boss_killed() else 0
        reward = ((health_life + stage_checkpoint + ex_skills) / 3) + boss_killed - ((self.total_play_time / (60 * 60)) * 0.5)


        # Check if done or truncated
        done = self.game_finished()
        truncated = self.check_if_truncated()

        # Update total reward
        self._total_reward += (reward - self._prev_reward)
        self._prev_reward = reward

        # Update best_stage and best_checkpoint
        if self.curr_stage < 17 and\
            (self.curr_stage, self.curr_checkpoint) > (self.best_stage, self.best_checkpoint):
                (self.best_stage, self.best_checkpoint) = (self.curr_stage, self.curr_checkpoint)

        info = {
            "health": self.health,
            "life_count": self.life_count,
            "total_play_time": self.total_play_time,
            "total_rewards": self._total_reward,
            "current_stage": self.curr_stage,
            "current_checkpoint": self.curr_checkpoint,
            "best_stage": self.best_stage,
            "best_checkpoint": self.best_checkpoint
        }

        self._step += 1

        return observation, reward, done, truncated, info
        
    def check_boss_killed(self) -> bool:
        return self.boss_health == 0 and\
                (self.curr_checkpoint in BOSS_LOCATION and self.curr_stage in BOSS_LOCATION[self.curr_checkpoint] or\
                 (self.curr_stage == 16 and self.final_subbosses_room in FINAL_STAGE_BOSSES[self.curr_checkpoint]))
    
    def check_if_truncated(self) -> bool:
        return (self.max_episode_steps is not None and 
                (self._step >= self.max_episode_steps)) or\
                self.curr_stage < 17 and self.checkpoint_time > 60 or\
                self.same_pos_count > 30 * (60 / 1 + self.frameskip) or\
                (self.life_count <= 0 and self.health <= 0)

    
    def game_finished(self) -> bool:
        # Return if game completed successfully
        return self.curr_stage == 16 and self.curr_checkpoint == 9 and\
                self.final_subbosses_room == 7 and self.boss_health == 0


    def reset(self, seed=None, options=None):
        # Reset game
        self.gba.core.reset()
        # Load initial state
        if self._initial_state is not None:
            self.gba.core.load_raw_state(self._initial_state)

            # not sure what the best solution is here:
            # 1. don't run_frame after resetting the state, will lead to the old frame still being rendered
            # 2. run_frame after resetting the state, offsetting the savestate by one frame
            self.gba.core.run_frame()

        self._get_game_data()
        self.total_play_time = 0
        self.same_pos_count = 0

        info = {
            "health": self.health,
            "life_count": self.life_count,
            "total_play_time": self.total_play_time,
            "total_rewards": self._total_reward,
            "current_stage": self.curr_stage,
            "current_checkpoint": self.curr_checkpoint
        }

        self._total_reward = 0
        self._step = 0
        
        observation = self._get_observation()
        return observation, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return
        
        img = self._framebuffer.to_pil().convert("RGB")
        if self.obs_type == "grayscale":
            img = img.convert("L")
        
        if self.render_mode == "human":
            if "pygame" not in sys.modules:
                raise RuntimeError(
                    "pygame is not installed, run `pip install pygame`"
                ) from e

            if self._screen is None:
                pygame.init()
                pygame.display.init()
                self._screen = pygame.display.set_mode(
                    self.gba.core.desired_video_dimensions()
                )
            if self._clock is None:
                self._clock = pygame.time.Clock()

            surf = _pil_image_to_pygame(img)
            self._screen.fill((0, 0, 0))
            self._screen.blit(surf, (0, 0))

            effective_fps = self.metadata["render_fps"]
            if self.frameskip:
                if isinstance(self.frameskip, tuple):
                    # average FPS is close enough
                    effective_fps /= (self.frameskip[0] + self.frameskip[1]) / 2 + 1
                else:
                    effective_fps /= self.frameskip + 1

            pygame.event.pump()
            self._clock.tick(effective_fps)
            pygame.display.flip()
        else:  # self.render_mode == "rgb_array"
            return np.array(img)

    def close(self):
        if self._screen is not None:
            if "pygame" not in sys.modules:
                pygame.display.quit()
                pygame.quit()

# Create callable that works for GymNE from EvoTroch 
def make_RMZ3Env(
        gba_rom: str | Path,
        gba_sav: str | Path,
        render_mode: Literal["human", "rgb_array"] | None = "human",
        frameskip: int | tuple[int, int] | tuple[int, int, int] = 0,
        mgba_silence: bool = False,
        to_resize: bool = True,
        to_grayscale: bool = False,
        record: bool = False,
        record_path: str | Path = None,
        **kwargs
        ) -> Callable[[], Env]:
    env = RMZ3Env(gba_rom = gba_rom,
                   gba_sav = gba_sav,
                   render_mode = render_mode,
                   frameskip = frameskip,
                   mgba_silence = mgba_silence,
                   **kwargs)
    if record and record_path is not None:
        env = RecordVideo(env, video_folder = record_path)
    if to_resize:
        env = ResizeObservation(env, (120, 80))
    if to_grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    return env
