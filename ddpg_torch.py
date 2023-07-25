import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ddpg_common import OUActionNoise, ReplayBuffer

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="checkpoints"):
        super(CriticNetwork, self).__init__()

        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        try:
            os.mkdir(chkpt_dir)
        except FileExistsError:
            pass
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}.critic.ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))

        state_action_value = T.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print(f"saving {self.checkpoint_file}")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f"loading {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="checkpoints"):
        super(ActorNetwork, self).__init__()

        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        try:
            os.mkdir(chkpt_dir)
        except FileExistsError:
            pass
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}.actor.ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        f3 = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.mu(state_value)
        state_value = T.tanh(state_value)

        return state_value

    def save_checkpoint(self):
        print(f"saving {self.checkpoint_file}")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f"loading {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions, 'actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions, 'target_actor')

        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                  n_actions, 'critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                  n_actions, 'critic_actor')

        self.noise = OUActionNoise(np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            # we don't have enough samples to learn yet
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        done = T.tensor(done, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

        target_actions = self.target_actor.forward(new_state)
        target_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * target_critic_value[j] * done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_parameters = self.actor.named_parameters()
        critic_parameters = self.critic.named_parameters()
        target_actor_parameters = self.target_actor.named_parameters()
        target_critic_parameters = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_parameters)
        critic_state_dict = dict(critic_parameters)
        target_actor_state_dict = dict(target_actor_parameters)
        target_critic_state_dict = dict(target_critic_parameters)

        for name in actor_state_dict:
            actor_state_dict[name] = tau     * actor_state_dict[name].clone() + \
                                     (1-tau) * target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        for name in critic_state_dict:
            critic_state_dict[name] = tau    * critic_state_dict[name].clone() + \
                                     (1-tau) * target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()



