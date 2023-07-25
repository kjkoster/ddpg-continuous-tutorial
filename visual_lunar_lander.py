import gym
import time
import numpy as np
from ddpg_torch import Agent

env = gym.make('LunarLanderContinuous-v2', render_mode="human")

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)
agent.load_models()

np.random.seed(0)

score_history = []
for episode in range(10000):
    start = time.time()
    done = False
    truncated = False
    score = 0
    observation, info = env.reset()

    while not (done or truncated):
        action = agent.choose_action(observation)
        observation, reward, truncated, done, info = env.step(action)
        score += reward
        env.render()

    score_history.append(score)
    print(f"episode {episode}: score {score:.2f}, 100 game average {np.mean(score_history[-100:]):.2f} took {time.time() - start:.1f} seconds")


