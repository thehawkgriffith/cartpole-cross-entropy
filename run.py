import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple

# hyperparameters
BATCH_SIZE = 32
PERCENTILE = 80

# namedtuple data structure to store episodes and 
# corresponding rewards
Episode = namedtuple('Episode', 
                     field_names=['step', 'reward'])
EpisodeStep = namedtuple('EpisodeStep', 
                         field_names=['observation', 
                                      'action'])


# model
class Agent(nn.Module):
    
    def __init__(self, input_dims, hidden, output_dims,
                 activ=nn.ReLU()):
        super(Agent, self).__init__()
        self.mod = nn.Sequential(nn.Linear(input_dims,
                                           hidden),
                      activ,
                      nn.Linear(hidden, output_dims))
    
    def forward(self, X):
        return self.mod(X)


# generator of batches
def iterate_batches(env, agent, batch_sz):
    batch = []
    sm = nn.Softmax(dim=1)
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    while True:
        obs = torch.FloatTensor([obs])
        act = agent(obs)
        act_p = sm(act).data.numpy()[0]
        action = np.random.choice(len(act_p), p=act_p)
        p_obs = obs[0].data.numpy()
        obs, r, done, _ = env.step(action)
        episode_reward += r
        episode_steps.append(EpisodeStep(p_obs, action))
        if done:
            batch.append(Episode(episode_steps
                                 , episode_reward))
            episode_reward = 0.0
            episode_steps = []
            obs = env.reset()
            if len(batch) == batch_sz:
                yield(batch)
                batch = []         

# filter samples based on percentile     
def filter_batch(batch, percentile):
    train_acts = []
    train_obs = []
    rewards = [example.reward for example in batch]
    reward_mean = np.mean(rewards)
    reward_bound = np.percentile(rewards, percentile)
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation,
                             example.step))
        train_acts.extend(map(lambda step: step.action,
                             example.step))
    train_acts_v = torch.LongTensor(train_acts)
    train_obs_v = torch.FloatTensor(train_obs)
    return train_acts_v, train_obs_v, reward_bound, reward_mean

# render function
def play_one(env, agent):
    obs = env.reset()
    done = False
    sm = nn.Softmax(dim=1)
    while not done:
        obs = torch.FloatTensor([obs])
        act = sm(agent(obs)).data.numpy()[0]
        action = np.argmax(act)
        obs, _, done, _ = env.step(action)
        env.render()
    env.close()

# main
env = gym.make('CartPole-v0')
agent = Agent(4, 128, 2)
objective = nn.CrossEntropyLoss()
optim = torch.optim.Adam(agent.parameters(), lr=0.01)

#training
for iter_no, batch in enumerate(iterate_batches(env, agent, BATCH_SIZE)):
    train_acts, train_obs, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
    optim.zero_grad()
    acts = agent(train_obs)
    loss = objective(acts, train_acts)
    lossa = loss.data.numpy()
    print("Epoch: %d  Loss: %.3f  Reward Mean: %.2f  Reward Bound: %.2f"%(iter_no, lossa, reward_mean, reward_bound))
    loss.backward()
    optim.step()
    if reward_mean > 199:
        break

# visualisation
play_one(env, agent)