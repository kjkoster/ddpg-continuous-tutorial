import gym
import time
import torch as T
import numpy as np
from ddpg_torch import Agent
import matplotlib.pyplot as plt

print(f"training on device {'cuda:0' if T.cuda.is_available() else 'cpu'}")

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

game='LunarLanderContinuous-v2'
env = gym.make(game)

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

# try loading checkpoint files, if they exist, so we can continue learning on
# those. If no files exist, we just start fresh.
try:
    agent.load_models(f"checkpoints/{game}")
except FileNotFoundError:
    pass

score_history = []
for episode in range(1000):
    start = time.time()
    done = False
    truncated = False
    score = 0
    iterations = 0
    observation, info = env.reset()

    while not (done or truncated):
        action = agent.choose_action(observation)
        new_state, reward, done, truncated, info = env.step(action)
        agent.remember(observation, action, reward, new_state, int(done or truncated))
        agent.learn()
        score += reward
        iterations += 1
        observation = new_state

    # punish the hovering behaviour by making time a cost
    score -= iterations

    score_history.append(score)
    print(f"episode {episode}: score {score:.2f}, 100 game average {np.mean(score_history[-100:]):.2f}, took {time.time() - start:.1f} seconds for {iterations} iterations, done {done}, truncated {truncated}")
    if (episode + 1) % 25 == 0:
        agent.save_models(f"checkpoints/{game}")
        plotLearning(score_history, f"{game}.png", window=100)

