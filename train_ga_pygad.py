from CNNLSTM import CNNLSTM
from ga import GA
import torch
from gymnasium.vector import AsyncVectorEnv
from RMZ3Env import RMZ3Env
from pygba import PyGBA
import mgba.log

mgba.log.silence()

# Create environment
def make_env(id):
    gba = PyGBA.load("./GBA_ROM/Rockman Zero 3 (Japan).gba", 
                    "./GBA_ROM/Rockman Zero 3 (Japan).sav")
    env = RMZ3Env(gba=gba, render_mode="rgb_array", frameskip=5, rank=id)
    return env

def main():
    # Create vectorized environment
    envs = AsyncVectorEnv([lambda: make_env(i) for i in range(16)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create and train the model using Genetic Algorithm (GA)
    model = CNNLSTM(envs.observation_space.shape, len(envs.action_space.shape)).to(device)
    ga = GA(model, envs, device)
    ga.max_steps = 20 * 60 * 60
    ga.train()

if __name__ == "__main__":
    main()
