import gym
from Model.class_model import  BC_agent, TD3_Agent
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
agent.init_bc("./model_save/bc/bc2_"+args.task_name+"_80.pt")
#agent.init_pi("./model_save/bc_wq/bc_wq_halfcheetah-random-v2_600__123.pt")

dataset = d4rl_dataset(env.unwrapped)

maximum_step = 100000
local_step = 0
eval_period = 5
episode_step = 0
n_random = 1000
cql = True
#====cql====

while local_step <=maximum_step:
  state = env.reset()

  batch = dataset.get_data()
  local_step += 1
  agent.train_Only_Q(batch,cql=cql)

  if local_step % 1000 == 999:
      batch = dataset.get_data()
      q1,q2 = agent.test_q(batch)
      print("[local_step] :",local_step+1, "Q1 : ",sum(q1)/batch[0].shape[0],"Q2 : ",sum(q2)/batch[0].shape[0])

  if local_step % 20000 == 19999:
    torch.save({'q1': agent.q1.state_dict(),
                'q2': agent.q2.state_dict(),
                }, "./model_save/bc_q/bc2_"+args.task_name+"cql"+str(cql)+"_"+ str(local_step + 1) + ".pt")
