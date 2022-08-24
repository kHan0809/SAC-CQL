import gym
from Model.class_model import  BC_agent
from Utils.arguments import get_args
import numpy as np
import torch
import d4rl
from Utils.utils import d4rl_dataset


args = get_args()

# env = gym.make("InvertedPendulum-v2")
env = gym.make(args.task_name)
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
eval_num=5
#====cql====
n_train_step_per_epoch=1000


while local_step <=maximum_step:
  state = env.reset()
  for step in range(n_train_step_per_epoch):
    batch = dataset.get_data()
    local_step += 1
    agent.train_bc(batch)
  episode_step += 1

  #====Eval====
  if episode_step % eval_period == 0:
    epi_return = []  # 나중에 success까지 포함해야할듯
    for eval_epi in range(eval_num):
      state = env.reset()
      total_reward = 0
      for step in range(epi_length):
        # if eval_epi == (eval_num-1):
        #     env.render()
        action = agent.select_action(state.reshape([1,-1]),eval=True)
        state, rwd, done, _ = env.step(action * action_max)
        total_reward += rwd
        if done:
          break
      epi_return.append(total_reward)
    print("==================[Eval]====================")
    print("Epi : ", episode_step)
    print("Mean return  : ", np.mean(epi_return), "Min return", np.min(epi_return), "Max return", np.max(epi_return))

  if episode_step % 20 == 19:
    torch.save({'policy': agent.bc.state_dict(),
                'log_alpha' : agent.log_alpha,
                }, "./model_save/bc/bc_"+args.task_name+"_"+str(episode_step + 1) + ".pt")

#medium-expert
# [EPI5] : 4465.38
# [EPI10] : 5156.28
# [EPI15] : 3261.79
# [EPI20] : 288.24
# [EPI25] : 5082.63
# [EPI30] : 5290.28
# [EPI35] : 4970.37
# [EPI40] : 4963.78
# [EPI45] : 5096.16
# [EPI50] : 5127.29
# [EPI55] : 5179.81
# [EPI60] : 5064.50
# [EPI65] : 5053.17
# [EPI70] : 10902.59
# [EPI75] : 10634.01
# [EPI80] : 6053.93
# [EPI85] : 5047.37
# [EPI90] : 5085.35
# [EPI95] : 5136.49

# medium
# [EPI5] : 4557.32
# [EPI10] : 1872.80
# [EPI15] : 4787.22
# [EPI20] : 5092.55
# [EPI25] : 4893.88
# [EPI30] : 5184.68
# [EPI35] : 4974.44
# [EPI40] : 4915.95
# [EPI45] : 4935.28
# [EPI50] : 4872.44

#expert
# [EPI5] : 336.94
# [EPI10] : 310.75
# [EPI15] : 7859.40
# [EPI20] : 3642.10
# [EPI25] : 10856.15
# [EPI30] : 10611.11
# [EPI35] : 10951.73
# [EPI40] : 10701.42
# [EPI45] : 10801.44
# [EPI50] : 11013.17
# [EPI55] : 11044.47
# [EPI60] : 10912.29
# [EPI65] : 11053.25
# [EPI70] : 11056.16
# [EPI75] : 11043.82
# [EPI80] : 11216.14
# [EPI85] : 10810.40
# [EPI90] : 11004.27