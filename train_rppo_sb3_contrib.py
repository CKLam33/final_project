import mgba.log
from pathlib import Path
from pygba import PyGBA
from make_env import RMZ3Env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO

mgba.log.silence()

def make_env(id, seed = 0):
    def _init():
        frame_path = Path("./frames")
        frame_path.mkdir(exist_ok=True, parents=True)
        gba = PyGBA.load("./GBA_ROM/Rockman Zero 3 (Japan).gba",
                         "./GBA_ROM/Rockman Zero 3 (Japan).sav")
        env = RMZ3Env(gba = gba,
                        render_mode = "rgb_array",
                        frameskip = 5,
                        frame_save = True,
                        rank = id,
                        frame_path = "./frames",
                        frame_save_freq = 100)
        return env
    return _init

envs = [make_env(i) for i in range(32)]

envs = SubprocVecEnv(envs, start_method="fork")

agent = RecurrentPPO("CnnLstmPolicy",
                     envs,
                     n_steps=16,
                     batch_size=2000,
                     learning_rate=1e-4,
                     ent_coef = 0.001,
                     target_kl = 1e-4,
                     tensorboard_log="./log",
                     device="auto")

agent.learn(1600000, progress_bar=True)
