import gym
import time
import numpy as np
from ddpg_torch import Agent

game='BipedalWalker-v3'
env = gym.make(game, render_mode="human")

agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[24], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=4)
agent.load_models(f"checkpoints/{game}")

score_history = []
for episode in range(10000):
    start = time.time()
    done = False
    truncated = False
    score = 0
    iterations = 0
    observation, info = env.reset()

    while not (done or truncated):
        action = agent.choose_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        score += reward
        iterations += 1
        env.render()

    score_history.append(score)
    delta_t = time.time() - start
    print(f"episode {episode}: score {score:.2f}, 100 game average {np.mean(score_history[-100:]):.2f}, took {delta_t:.1f} seconds for {iterations} iterations, {iterations/delta_t:.1f} iterations per second, done {done}, truncated {truncated}")

