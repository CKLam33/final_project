import torch
import torch.nn as nn
import numpy as np
import pygad
import pygad.torchga as torchga
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm

class GA():
    
    def __init__(self, model: nn.Module, envs: AsyncVectorEnv, device: torch.device):
        self.model = model.to(device)
        self.env = envs
        self.device = device
        self.max_steps = 0
        
        # Initialize TorchGA with population size
        self.torch_ga = torchga.TorchGA(model=self.model, 
                                        num_solutions = self.env.num_envs)
        
        self.population = self.torch_ga.population_weights  # Get initial population

    def set_max_steps(self, max_steps: int = 1000):
        self.max_steps = max_steps

    def fitness(self, ga_instance, solution, solution_idx):
        # Load weights into model
        self._model_weights = torchga.model_weights_as_dict(
            model=self.model,
            weights_vector=solution
        )
        self.model.load_state_dict(self._model_weights)

        # Get observations from all environments
        obs, _ = self.env.reset()
        total_rewards = np.zeros(self.env.num_envs)
        hidden = None
        dones = np.zeros(self.env.num_envs, dtype=bool)
        steps = 0

        # Take action and collect rewards
        while (steps < self.max_steps) or (not np.all(dones)):
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.uint8) / 255
            with torch.no_grad():
                action_probs, hidden = self.model(obs_tensor, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            actions = torch.argmax(action_probs, dim=1).detach().cpu().numpy()
            obs, rewards, terminateds, truncateds, _ = self.env.step(actions)
            total_rewards += rewards
            dones = terminateds | truncateds
            steps += 1

        return np.mean(total_rewards)  # Average reward across environments


# num_generations: int,
# num_parents_mating: int,
# fitness_func: (ga_instance: Any, solution: Any, solution_idx: Any) -> floating[Any],
# fitness_batch_size: Any | None = None,
# initial_population: Any | None = None,
# sol_per_pop: Any | None = None,
# num_genes: Any | None = None,
# init_range_low: int = -4,
# init_range_high: int = 4,
# gene_type: type[float] = float,
# parent_selection_type: str = "sss",
# keep_parents: int = -1,
# keep_elitism: int = 1,
# K_tournament: int = 3,
# crossover_type: str = "single_point",
# crossover_probability: Any | None = None,
# mutation_type: str = "random",
# mutation_probability: Any | None = None,
# mutation_by_replacement: bool = False,
# mutation_percent_genes: str = 'default',
# mutation_num_genes: Any | None = None,
# random_mutation_min_val: float = -1,
# random_mutation_max_val: float = 1,
# gene_space: Any | None = None,
# allow_duplicate_genes: bool = True,
# on_start: Any | None = None,
# on_fitness: Any | None = None,
# on_parents: Any | None = None,
# on_crossover: Any | None = None,
# on_mutation: Any | None = None,
# on_generation: Any | None = None,
# on_stop: Any | None = None,
# save_best_solutions: bool = False,
# save_solutions: bool = False,
# suppress_warnings: bool = False,
# stop_criteria: Any | None = None,
# parallel_processing: Any | None = None,
# random_seed: Any | None = None,
# logger: Any | None = None
    
    def train(self,
              num_generations: int = 1000,
              mutation_percent: int = 5,
              parent_selection_type: str = "rank",
              crossover_type: str = "uniform",
              mutation_type: str = "random",
              keep_elitism: int = 1,
              suppress_warnings: bool = True):
              
        
        ga_instance = pygad.GA(
            num_generations = num_generations,
            num_parents_mating = max(4, self.env.num_envs//4),
            initial_population = self.population,
            fitness_func = self.fitness,
            mutation_percent_genes = mutation_percent,
            parent_selection_type = parent_selection_type,
            crossover_type = crossover_type,
            mutation_type = mutation_type,
            keep_elitism = keep_elitism,
            suppress_warnings = suppress_warnings
        )

        with tqdm(total = ga_instance.num_generations, desc = "Training Progress") as pbar:
            def callback():
                pbar.update(1)
            ga_instance.on_generation = callback
            ga_instance.run()


