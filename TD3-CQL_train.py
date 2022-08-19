import gym
from Model.class_model import TD3_Agent
from Utils.arguments import get_args
import numpy as np
import torch
import d4rl
from Utils.utils import d4rl_dataset


args = get_args()

# env = gym.make("InvertedPendulum-v2")
env = gym.make("halfcheetah-medium-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
epi_length = env.spec.max_episode_steps


agent = TD3_Agent(state_dim,action_dim,args)
dataset = d4rl_dataset(env.unwrapped)

maximum_step = 1000000
local_step = 0
eval_period = 5
episode_step = 0
n_random = 1000
#====cql====
n_train_step_per_epoch=1000


while local_step <=maximum_step:
  state = env.reset()
  for step in range(n_train_step_per_epoch):
    batch = dataset.get_data()
    local_step += 1
    agent.train_off(batch,cql=True)
  episode_step += 1

  # Evaluation
  if episode_step % eval_period == 0:
    state = env.reset()
    total_reward = 0
    for step in range(epi_length):
      # env.render()
      action = agent.select_action(state.reshape([1,-1]),eval=True)
      next_state, rwd, done, _ = env.step(action*action_max)
      total_reward += rwd
      state = next_state
      if done:
        break
    print("[EPI%d] : %.2f"%(episode_step, total_reward))

  # if episode_step % 200 == 199:
  #   torch.save({'policy': agent.pi.state_dict(),
  #               'Q_val1': agent.q1.state_dict(),
  #               'Q_val2': agent.q2.state_dict()
  #               }, "./model_save/sac-cql/SAC-CQL_model_" + str(episode_step + 1) + ".pt")

#
# [EPI5] : -197.48
# [EPI10] : -698.14
# [EPI15] : -869.52
# [EPI20] : -270.97
# [EPI25] : -298.27
# [EPI30] : -346.55
# [EPI35] : -593.65
# [EPI40] : -3.49
# [EPI45] : 4984.58
# [EPI50] : 5122.06
# [EPI55] : 5255.96
# [EPI60] : 2973.60
# [EPI65] : 5207.06
# [EPI70] : 5285.03
