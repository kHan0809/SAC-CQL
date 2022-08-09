import gym
from Model.class_model import SAC_Agent
from Utils.arguments import get_args
import numpy as np
import torch

args = get_args()

# env = gym.make("InvertedPendulum-v2")
env = gym.make("HalfCheetah-v2")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
epi_length = env.spec.max_episode_steps

agent = SAC_Agent(state_dim,action_dim,args)

maximum_step = 1000000
local_step = 0
eval_period = 5
episode_step = 0
n_random = 1000


while local_step <=maximum_step:
  state = env.reset()
  for step in range(epi_length):
    if local_step > n_random:
      action = agent.select_action(state.reshape([1,-1]))
    else:
      action = np.random.uniform(-1.,1.,action_dim)

    next_state ,rwd, done, _ = env.step(action*action_max)
    local_step += 1

    if done==True and step == epi_length-1:
      terminal = False
    else:
      terminal = done
    agent.buffer.store_sample(state, action, rwd, next_state, terminal)

    agent.train()
    state = next_state
    if done:
      break
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

  if episode_step % 200 == 199:
    torch.save({'policy': agent.pi.state_dict(),
                'Q_val1': agent.q1.state_dict(),
                'Q_val2': agent.q2.state_dict()
                }, "./model_save/sac/SAC_model_" + str(episode_step + 1) + ".pt")




# [EPI5] : -25.11
# [EPI10] : -97.38
# [EPI15] : -61.14
# [EPI20] : -52.46
# [EPI25] : 271.74
# [EPI30] : 1021.35
# [EPI35] : 2142.83
# [EPI40] : 2767.47
# [EPI45] : 2699.34
# [EPI50] : 2628.31
# [EPI55] : 3243.12
# [EPI60] : 3412.57
# [EPI65] : 3412.31
# [EPI70] : 3515.16

