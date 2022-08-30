import gym
from Model.class_model import SAC_CQL_Agent
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


agent = SAC_CQL_Agent(state_dim,action_dim,args)
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


    batch = dataset.get_data()
    q1, q2 = agent.test_q(batch)
    print("[local_step] :", local_step + 1, "Q1 : ", sum(q1) / batch[0].shape[0], "Q2 : ",sum(q2) / batch[0].shape[0])

  # if episode_step % 200 == 199:
  #   torch.save({'policy': agent.pi.state_dict(),
  #               'Q_val1': agent.q1.state_dict(),
  #               'Q_val2': agent.q2.state_dict()
  #               }, "./model_save/sac-cql/SAC-CQL_model_" + str(episode_step + 1) + ".pt")



# [EPI5] : -368.18
# [EPI10] : -212.60
# [EPI15] : -355.29
# [EPI20] : 2700.51
# [EPI25] : 1615.43
# [EPI30] : 4730.36
# [EPI35] : 930.74
# [EPI40] : 1499.38
# [EPI45] : 1383.29
# [EPI50] : 5314.87
# [EPI55] : 5234.07
# [EPI60] : 5309.81
# [EPI65] : 5552.11
# [EPI70] : 5322.81
# [EPI75] : 5277.82
# [EPI80] : 5424.34
# [EPI85] : 5599.71
# [EPI90] : 5443.05
# [EPI95] : 5388.76
# [EPI100] : 5479.21
# [EPI105] : 5529.77
# [EPI110] : 5516.31
# [EPI115] : 5377.74
# [EPI120] : 5676.22
# [EPI125] : 5560.77
# [EPI130] : 5694.33
# [EPI135] : 5531.21
# [EPI140] : 5646.63
# [EPI145] : 5683.68
# [EPI150] : 5424.42
# [EPI155] : 5642.67
# [EPI160] : 5710.20
# [EPI165] : 5643.17
# [EPI170] : 5603.17
# [EPI175] : 5525.25
# [EPI180] : 5513.57
# [EPI185] : 5768.60
# [EPI190] : 5591.29
# [EPI195] : 5706.61
# [EPI200] : 5506.01
# [EPI205] : 5773.31
# [EPI210] : 5528.96
# [EPI215] : 5680.78
# [EPI220] : 5478.72
# [EPI225] : 5698.83
# [EPI230] : 5765.45
# [EPI235] : 5860.26
# [EPI240] : 5551.79
# [EPI245] : 5519.28
# [EPI250] : 5571.65
# [EPI255] : 5611.70
# [EPI260] : 5772.80
# [EPI265] : 5659.49
# [EPI270] : 5636.44
# [EPI275] : 5571.46

#Medium
# [Return mean] :  4770.215598305823
# [EPI5] : -71.79
# [local_step] : 5001 Q1 :  tensor(96.8751, device='cuda:0') Q2 :  tensor(96.9181, device='cuda:0')
# [EPI10] : -729.93
# [local_step] : 10001 Q1 :  tensor(181.2735, device='cuda:0') Q2 :  tensor(181.3844, device='cuda:0')
# [EPI15] : -168.59
# [local_step] : 15001 Q1 :  tensor(236.1676, device='cuda:0') Q2 :  tensor(236.4753, device='cuda:0')
# [EPI20] : -164.32
# [local_step] : 20001 Q1 :  tensor(273.3922, device='cuda:0') Q2 :  tensor(273.2821, device='cuda:0')
# [EPI25] : 175.14
# [local_step] : 25001 Q1 :  tensor(305.7065, device='cuda:0') Q2 :  tensor(305.5658, device='cuda:0')
# [EPI30] : 401.15
# [local_step] : 30001 Q1 :  tensor(320.3622, device='cuda:0') Q2 :  tensor(320.5507, device='cuda:0')
# [EPI35] : 4144.52
# [local_step] : 35001 Q1 :  tensor(335.2350, device='cuda:0') Q2 :  tensor(334.9603, device='cuda:0')
# [EPI40] : 5085.33
# [local_step] : 40001 Q1 :  tensor(345.1120, device='cuda:0') Q2 :  tensor(345.1349, device='cuda:0')
# [EPI45] : 5330.91
# [local_step] : 45001 Q1 :  tensor(352.8109, device='cuda:0') Q2 :  tensor(352.9360, device='cuda:0')
# [EPI50] : 4970.47
# [local_step] : 50001 Q1 :  tensor(353.9359, device='cuda:0') Q2 :  tensor(354.0067, device='cuda:0')
# [EPI55] : 5429.06
# [local_step] : 55001 Q1 :  tensor(358.9911, device='cuda:0') Q2 :  tensor(359.0433, device='cuda:0')
# [EPI60] : 5298.67
# [local_step] : 60001 Q1 :  tensor(361.8062, device='cuda:0') Q2 :  tensor(361.6589, device='cuda:0')
# [EPI65] : 5433.67
# [local_step] : 65001 Q1 :  tensor(365.3981, device='cuda:0') Q2 :  tensor(365.3272, device='cuda:0')
# [EPI70] : 5475.62
# [local_step] : 70001 Q1 :  tensor(362.9278, device='cuda:0') Q2 :  tensor(362.8710, device='cuda:0')
# [EPI75] : 5424.34
# [local_step] : 75001 Q1 :  tensor(361.0743, device='cuda:0') Q2 :  tensor(361.0407, device='cuda:0')
# [EPI80] : 5431.59
# [local_step] : 80001 Q1 :  tensor(363.2005, device='cuda:0') Q2 :  tensor(363.0135, device='cuda:0')
# [EPI85] : 5483.38
# [local_step] : 85001 Q1 :  tensor(363.8866, device='cuda:0') Q2 :  tensor(363.8287, device='cuda:0')
# [EPI90] : 5472.43
# [local_step] : 90001 Q1 :  tensor(363.6799, device='cuda:0') Q2 :  tensor(363.9665, device='cuda:0')
# [EPI95] : 5495.46
# [local_step] : 95001 Q1 :  tensor(357.1365, device='cuda:0') Q2 :  tensor(357.3423, device='cuda:0')
# [EPI100] : 5668.96
# [local_step] : 100001 Q1 :  tensor(364.9618, device='cuda:0') Q2 :  tensor(364.9307, device='cuda:0')
# [EPI105] : 5590.54
# [local_step] : 105001 Q1 :  tensor(361.3735, device='cuda:0') Q2 :  tensor(361.1693, device='cuda:0')
# [EPI110] : 5622.35
# [local_step] : 110001 Q1 :  tensor(362.1713, device='cuda:0') Q2 :  tensor(362.4529, device='cuda:0')