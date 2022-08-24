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
agent.init_bc("./model_save/bc/bc_"+args.task_name+"_100.pt")
agent.init_q("./model_save/bc_q/bc_"+args.task_name+"cqlTrue_"+"100000.pt")
#agent.init_bc("./model_save/bc_wq/bc_wq_halfcheetah-random-v2_600__123.pt")
#agent.init_q("./model_save/bc_q_test/bc_"+args.task_name+"cqlTrue_"+"100000.pt")
dataset = d4rl_dataset(env.unwrapped)



maximum_step = 1000000
local_step = 0
eval_period = 5
episode_step = 0
n_random = 1000
eval_num = 10
#====cql====
n_train_step_per_epoch=1000


while local_step <=maximum_step:
  state = env.reset()
  for step in range(n_train_step_per_epoch):
    batch = dataset.get_data()
    local_step += 1
    agent.train_weightedQ(batch)
  episode_step += 1

  # ====Eval====
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
                }, "./model_save/bc_wq/bc_wq_"+args.task_name+"_"+ str(episode_step + 1) + ".pt")


#medium-expert


#medium
# [EPI5] : 3729.93
# [EPI10] : 4683.90
# [EPI15] : 4880.29
# [EPI20] : 4995.83
# [EPI25] : 5152.09
# [EPI30] : 5089.73
# [EPI35] : 5026.14
# [EPI40] : 5172.80
# [EPI45] : 5145.51
# [EPI50] : 4920.03
# [EPI55] : 5117.37
# [EPI60] : 5078.40
# [EPI65] : 5180.81
# [EPI70] : 5037.74

#expert
# [EPI5] : 649.84
# [EPI10] : 2151.19
# [EPI15] : 3860.30
# [EPI20] : 492.21
# [EPI25] : 5051.66
# [EPI30] : 10042.76
# [EPI35] : 8611.61
# [EPI40] : 6616.41
# [EPI45] : 10948.23
# [EPI50] : 10912.86
# [EPI55] : 10570.31
# [EPI60] : 10801.76
# [EPI65] : 11163.10
# [EPI70] : 11020.28
# [EPI75] : 11418.70
# [EPI80] : 11332.31
# [EPI85] : 11056.12
# [EPI90] : 11178.14
# [EPI95] : 11400.97
# [EPI100] : 11323.55