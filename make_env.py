import numpy as np
# import torch
from pygba import PyGBA, PyGBAEnv, GameWrapper
from stable_baselines3.common.atari_wrappers import WarpFrame
# from stable_baselines3.common.utils import set_random_seed

class PyGBAEnvWithClipReward(PyGBAEnv):
    def __init__(self, *args, reward_clipping: bool = False, pos_scale=5.0, neg_scale=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_clipping = reward_clipping
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale

    def clip_reward(self, reward: float) -> float:
        """Clip the reward using a hyperbolic tangent function."""
        reward = np.tanh(reward)
        if reward > 0:
            reward *= self.pos_scale
        else:
            reward *= self.neg_scale
        return reward

    def step(self, action_id):
        """Override the step method to apply clipped rewards."""
        observation, reward, done, truncated, info = super().step(action_id)
        if self.reward_clipping:
            reward = self.clip_reward(reward)  # Apply the clipping function to the reward
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        if seed is None:
            seed = np.random.randint(2 ** 32 - 1)
        super().reset(seed=seed)
        info = {}
        self._total_reward = 0
        self._step = 0
        self.gba.core.reset()
        if self._initial_state is not None:
            self.gba.core.load_raw_state(self._initial_state)
            self.gba.core.run_frame()
        
        observation = self._get_observation()
        
        if self.game_wrapper is not None:
            self.game_wrapper.reset(self.gba)
            info.update(self.game_wrapper.info(self.gba, observation))
        
        return observation, info  # Make sure to return both observation and info


def make_game_env(game: PyGBA, wrapper: GameWrapper, useWarpFrame: bool = True, seed:int = 0, width:int = 120, height: int = 80, **kwargs) -> PyGBAEnvWithClipReward:
    env = PyGBAEnvWithClipReward(game, wrapper, **kwargs)
    if useWarpFrame:
        env = WarpFrame(env, width = width, height = height)
        env.reset(seed = seed)
    return env