import sys
import random
from pathlib import Path

import numpy as np
import torch

if len(sys.argv) < 2:
    exit("Missing model file")

model_path = Path(sys.argv[1])
if not model_path.is_file():
    exit(f"Model file '{model_path}' not found")

from dqn_agent import DQNAgent
from tetris import Tetris, cv2


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()
env = Tetris()
render = cv2 is not None
if not render:
    print("OpenCV is not available; running without rendering.")
# pass a previously saved `.pt` model file
agent = DQNAgent(env.get_state_size(), modelFile=str(model_path))
done = False

while not done:
    next_states = {tuple(v): k for k, v in env.get_next_states().items()}
    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done = env.play(best_action[0], best_action[1], render=render)

print("Final score:", env.get_game_score())
env.close()
