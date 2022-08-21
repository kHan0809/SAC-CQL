import gym
from Model.class_model import TD3_Agent
from Utils.arguments import get_args
import numpy as np
import torch
import d4rl
from Utils.utils import d4rl_dataset

import matplotlib.pyplot as plt

args = get_args()

# env = gym.make("InvertedPendulum-v2")
env = gym.make("halfcheetah-medium-expert-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
epi_length = env.spec.max_episode_steps

agent = TD3_Agent(state_dim,action_dim,args)
agent.init_pi("./model_save/bc/bc_policy50.pt")
agent.init_q("./model_save/bc_q/bc_q_cql100000.pt")

dataset = d4rl_dataset(env.unwrapped)

maximum_step = 1000000
local_step = 0
eval_period = 5
episode_step = 0
#====cql====
n_train_step_per_epoch=1000

while local_step <=maximum_step:

  # Evaluation
  state = env.reset()
  total_reward = 0
  for step in range(epi_length):
    env.render()
    action = agent.select_action(state, eval=True)
    next_state, rwd, done, _ = env.step(action * action_max)
    total_reward += rwd
    state = next_state
    if done:
      break
  print("[EPI%d] : %.2f" % (episode_step, total_reward))

  state = env.reset()
  for step in range(5000):
    batch = dataset.get_data()
    local_step += 1
    agent.train_off(batch,cql=True)
  episode_step += 1

  # q1, q2 = agent.test_q(batch)
  # print("[local_step] :", local_step + 1, "Q1 : ", sum(q1) / batch[0].shape[0], "Q2 : ", sum(q2) / batch[0].shape[0])

  # # Evaluation
  # if episode_step % eval_period == 0:
  #   state = env.reset()
  #   total_reward = 0
  #   for step in range(epi_length):
  #     # env.render()
  #     action = agent.select_action(state,eval=True)
  #     next_state, rwd, done, _ = env.step(action*action_max)
  #     total_reward += rwd
  #     state = next_state
  #     if done:
  #       break
  #   print("[EPI%d] : %.2f"%(episode_step, total_reward))

  # if episode_step % 200 == 199:
  #   torch.save({'policy': agent.pi.state_dict(),
  #               'q1': agent.q1.state_dict(),
  #               'q2': agent.q2.state_dict()
  #               }, "./model_save/td-bc/td-bc_" + str(episode_step + 1) + ".pt")


# [EPI0] : 11265.05
# [EPI1] : 570.29
# [EPI2] : 3465.55
# [EPI3] : 10306.31
# [EPI4] : 42.15
# [EPI5] : 3454.02
# [EPI6] : 9508.15
# [EPI7] : 436.39
# [EPI8] : 9273.99
# [EPI9] : 9469.46
# [EPI10] : 9131.68
# [EPI11] : 10794.08
# [EPI12] : 8497.24
# [EPI13] : 9396.50