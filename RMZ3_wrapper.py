"""
Create game wrapper for Rockman Zero 3
"""

from typing import Any
import time
import numpy as np
from pygba import PyGBA, GameWrapper

class RockmanZero3Wrapper(GameWrapper):
    def __init__(self):
        self.start_time = time.time()
        self.health = 0
        self.life_count = 0
        self.stage = 0
        self.checkpoint = 0
        self.ingame_timer = 0
        self.total_play_time = 0

    def reward(self, gba: PyGBA, observation: np.ndarray) -> float:
        self.health = gba.read_u16(0x02037D04)
        self.life_count = gba.read_u16(0x02036F70)
        self.stage = gba.read_u16(0x0202FE60)
        self.checkpoint = gba.read_u16(0x0202FE62)
        self.ingame_timer = gba.read_u16(0x0202FE28) / 60  # Convert timer from frame to seconds
        
        # Calculate total play time in seconds
        self.total_play_time = time.time() - self.start_time
        
        # Reward calculation based on game state
        reward_health = self.health / 32  # Normalized health (0 to 1)
        reward_life = self.life_count / 9  # Normalized life count (0 to 9)
        
        # Number of stages passed and checkpoints based on their values
        reward_stage = self.stage / 16  # Normalized stage (1 to 16)
        reward_checkpoint = (self.checkpoint - 65281) / 7  # Normalized checkpoints (65281 to 65288)
        
        # Penalty based on timer and total play time
        penalty_ingame_timer = min(self.ingame_timer / 450, 1)  # Assuming a max penalty for over 2 minutes
        penalty_play_time = min(self.total_play_time / 7200, 1)  # Max penalty for over 2 hours

        # Total reward calculation
        total_reward = (reward_health + reward_life + reward_stage + reward_checkpoint) - (penalty_ingame_timer + penalty_play_time)

        return max(total_reward, 0)  # Ensure the reward is not negative

    def game_over(self, gba: PyGBA, observation: np.ndarray) -> bool:

        return self.life_count == 0 or self.total_play_time > 7200

    def reset(self, gba: PyGBA) -> None:
        self.start_time = time.time()
        self.health = 0
        self.life_count = 0
        self.stage = 0
        self.checkpoint = 0
        self.ingame_timer = 0
        self.total_play_time = 0

    def info(self, gba: PyGBA, observation: np.ndarray) -> dict[str, Any]:
        
        return {
            "health": self.health,
            "life_count": self.life_count,
            "stage": self.stage,
            "total_play_time": time.time() - self.start_time
        }