import gym
from Model.class_model import  BC_agent
from Utils.arguments import get_args
import numpy as np
import torch
import d4rl
from Utils.utils import d4rl_dataset


args = get_args()

# env = gym.make("InvertedPendulum-v2")
env = gym.make("halfcheetah-expert-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
epi_length = env.spec.max_episode_steps


agent = BC_agent(state_dim,action_dim,args)
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
    agent.train(batch)
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

  if episode_step % 5 == 4:
    torch.save({'policy': agent.bc.state_dict(),
                }, "./model_save/bc/bc_policy" + str(episode_step + 1) + ".pt")

# [EPI5] : 481.44
# [EPI10] : 278.78
# [EPI15] : 2403.08
# [EPI20] : 4210.49
# [EPI25] : 5056.42
# [EPI30] : 5108.28
# [EPI35] : 5224.96
# [EPI40] : 4810.20