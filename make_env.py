import sys
import time
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from pygba import PyGBA
import mgba.core
import mgba.image
import numpy as np

from pygba import PyGBA
from torch.utils.tensorboard import SummaryWriter

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    pass

from mgba.gba import GBA

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

def _pil_image_to_pygame(img):
    
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode).convert()

class RMZ3Env(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(
        self,
        gba: PyGBA,
        # raw_state: Any,
        obs_type: Literal["rgb", "grayscale"] = "rgb",
        frameskip: int | tuple[int, int] | tuple[int, int, int] = 0,
        frame_save: bool = True,
        rank: int = 0,
        frame_path: str | Path | None = None,
        frame_save_freq: int = 1,
        repeat_action_probability: float = 0.0,
        render_mode: Literal["human", "rgb_array"] | None = "human",
        reset_to_initial_state: bool = True,
        max_episode_steps: int | None = None,
        **kwargs,
    ):
        self.gba = gba
        self.obs_type = obs_type
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.arrow_keys = [None, "up", "down", "right", "left"]
        self.buttons = [None, "A", "B", "start", "L", "R"]

        # cartesian product of arrows and buttons, i.e. can press 1 arrow and 1 button at the same time
        self.actions = [(a, b) for a in self.arrow_keys for b in self.buttons]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Building the observation_space
        screen_size = self.gba.core.desired_video_dimensions()
        if obs_type == "rgb":
            screen_size += (3,)  # RGB channels
        else:  # grayscale
            screen_size += (1,)  # Single channel
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=screen_size, 
            dtype=np.uint8
        )

        self._framebuffer = mgba.image.Image(*self.gba.core.desired_video_dimensions())
        self.gba.core.set_video_buffer(self._framebuffer)  # need to reset after this

        self._screen = None
        self._clock = None

        # Saving frames
        self.frame_save = frame_save
        self.rank = rank
        self.frames_path = frame_path
        self.frame_save_freq = frame_save_freq

        self._total_reward = 0
        self._prev_reward = 0
        self._step = 0

        # Inintialize state
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

        self.health = 0
        self.life_count = 0
        self.stage = 0
        self.checkpoint = 0
        self.ingame_timer = 0
        self.prev_ingame_timer = 0
        self.start_time = 0
        self.total_play_time = 0

        # Reset the environment
        self.reset()

        self.writer = SummaryWriter(log_dir=f'logs/RockmanZero3_{self.rank:02d}')
    def get_action_by_id(self, action_id: int) -> tuple[Any, Any]:
        if action_id < 0 or action_id > len(self.actions):
            raise ValueError(f"action_id {action_id} is invalid")
        return self.actions[action_id]


    def get_action_id(self, arrow: str, button: str) -> int:
        action = (arrow, button)
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
        self.health = self.gba.read_u16(0x02037D04)
        self.life_count = self.gba.read_u16(0x02036F70)
        self.stage = self.gba.read_u16(0x0202FE60)
        self.checkpoint = self.gba.read_u16(0x0202FE62) - 65280
        self.ingame_timer = self.gba.read_u16(0x0202FE28) / 60
        if self.ingame_timer > self.prev_ingame_timer:
            self.ingame_timer = self.prev_ingame_timer

        info = {
            "norm_health": self.health / 16,
            "norm_life_count": self.life_count / 2,
            "norm_stage": self.stage / 16,
            "norm_checkpoint": self.checkpoint / 7,
            "ingame_timer": self.ingame_timer,
            "total_play_time": time.time() - self.start_time,
            "total_rewards": self._total_reward,
        }

        actions = self.get_action_by_id(action_id)
        actions = [KEY_MAP[a] for a in actions if a is not None]
        if np.random.random() > self.repeat_action_probability:
            self.gba.core.set_keys(*actions)

        if isinstance(self.frameskip, tuple):
            frameskip = np.random.randint(*self.frameskip)
        else:
            frameskip = self.frameskip

        for _ in range(frameskip + 1):
            self.gba.core.run_frame()
        observation = self._get_observation()

        if self.frame_save and self.frames_path is not None:
            # print(self._step)
            if (self._step + 1) % self.frame_save_freq == 0:
                # print("run")
                img = self._framebuffer.to_pil().convert("RGB")
                out_path = Path(self.frames_path) / f"{self.rank:02d}" / f"{self._step:06d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path)

        # Reward is calculated by adding up all the normalized values and subtracting the minimum normalized
        # ingame timer and total play time. This encourages the agent to focus on the most important aspects
        # of the game.
        reward = (info['norm_health'] + info['norm_life_count'] +
                  info['norm_stage'] + info['norm_checkpoint']) - \
                    (min(info['ingame_timer'] / ((3*60*60)/16), 1) * \
                      2 + min(info['total_play_time'] / (3*60*60), 1)) * 2

        # Check if done or truncated
        done = self.check_if_done()
        truncated = self.check_if_truncated()
        if self.max_episode_steps is not None:
            truncated = self._step >= self.max_episode_steps

        # Update total reward
        if self._step > 0:
            self._total_reward += (reward - self._prev_reward)
        self._prev_reward = reward

        self.writer.add_scalar('Total reward', self._total_reward, self._step)

        self._step += 1

        return observation, reward, done, truncated, info
    
    def check_if_truncated(self):
        return (self.max_episode_steps is not None and 
                (self._step + 1 >= self.max_episode_steps))
    
    def check_if_done(self):
        if self.game_over() or self.game_finished():
            return True
        return False
    
    def game_finished(self) -> bool:
        # Return if game completed successfully
        return self.stage == 16 and self.checkpoint == 65287 \
                and self.ingame_timer == self.prev_ingame_timer
    
    def game_over(self) -> bool:
        # Return if character has no life left or total play time has exceeded 3 hours
        # or ingame timer has exceeded 450 seconds
        # or total play time has exceeded 675 seconds
        return (self.life_count <= 0 and self.total_play_time > 0 and self.health <= 0) or \
                self.total_play_time > (3 * 60 * 60) or \
                self.ingame_timer > 450 or self.total_play_time / 16 > 675

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

        self.health = self.gba.read_u16(0x02037D04)
        self.life_count = self.gba.read_u16(0x02036F70)
        self.stage = self.gba.read_u16(0x0202FE60)
        self.checkpoint = self.gba.read_u16(0x0202FE62) - 65280
        self.ingame_timer = 0
        self.start_time = time.time()
        self.total_play_time = 0

        info = {
            "norm_health": self.health / 16,
            "norm_life_count": self.life_count / 2,
            "norm_stage": self.stage / 16,
            "norm_checkpoint": self.checkpoint / 7,
            "ingame_timer": self.ingame_timer,
            "total_play_time": time.time() - self.start_time,
            "total_rewards": self._total_reward,
        }

        self._total_reward = 0
        self._step = 0
        if self._initial_state is not None:
            self.gba.core.load_raw_state(self._initial_state)
            self.gba.core.run_frame()
        
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
        # self.writer.close()
        if self._screen is not None:
            if "pygame" not in sys.modules:
                pygame.display.quit()
                pygame.quit()