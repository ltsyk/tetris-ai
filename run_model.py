import sys
import random

import numpy as np
import torch

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris import Tetris


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()
env = Tetris()
# pass a previously saved `.pt` model file
agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[1])
done = False

while not done:
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1], render=True)

env.close()
