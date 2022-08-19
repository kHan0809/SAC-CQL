import gym
from Model.class_model import TD3_Agent
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

agent = TD3_Agent(state_dim,action_dim,args)

maximum_step = 1000000
local_step = 0
eval_period = 5
eval_num = 5
episode_step = 0
n_random = 10000


while local_step <=maximum_step:
  state = env.reset()
  for step in range(epi_length):
    if local_step > n_random:
      action = agent.select_action(state)
    else:
      action = env.action_space.sample() / action_max

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

  # =====Evaluation=====
  if episode_step % eval_period == 0:
    epi_return = [] #나중에 success까지 포함해야할듯
    for eval_epi in range(eval_num):
        state = env.reset()
        total_reward = 0
        for step in range(epi_length):
            # if eval_epi == (eval_num-1):
            #     env.render()
            action = agent.select_action(state,eval=True)
            state, rwd, done, _ = env.step(action*action_max)
            total_reward += rwd
            if done:
              break
        epi_return.append(total_reward)
    print("==================[Eval]====================")
    print("Epi : ", episode_step)
    print("Mean return  : ", np.mean(epi_return),"Min return",np.min(epi_return),"Max return",np.max(epi_return))

  # if episode_step % 200 == 199:
  #   torch.save({'policy': agent.pi.state_dict(),
  #               'Q_val1': agent.q1.state_dict(),
  #               'Q_val2': agent.q2.state_dict()
  #               }, "./model_save/sac/SAC_model_" + str(episode_step + 1) + ".pt")

# ==================[Eval]====================
# Epi :  5
# Mean return  :  -2.568387977862707 Min return -3.1863760415127254 Max return -2.0509764958222427
# ==================[Eval]====================
# Epi :  10
# Mean return  :  -2.5072706070118578 Min return -3.5181642234883346 Max return -1.8830656327823478
# ==================[Eval]====================
# Epi :  15
# Mean return  :  -403.12998080562903 Min return -419.3535532568008 Max return -368.73405203215196
# ==================[Eval]====================
# Epi :  20
# Mean return  :  -407.69526405781954 Min return -457.75570311549546 Max return -348.07168943639317
# ==================[Eval]====================
# Epi :  25
# Mean return  :  -236.89084464903468 Min return -435.4858305877823 Max return -18.70469344045217
# ==================[Eval]====================
# Epi :  30
# Mean return  :  529.3044212043258 Min return 396.0884093995468 Max return 666.3817268582221
# ==================[Eval]====================
# Epi :  35
# Mean return  :  1202.2530445937896 Min return 1087.3663818109715 Max return 1302.1469539734132
# ==================[Eval]====================
# Epi :  40
# Mean return  :  2034.5094882080223 Min return 1917.1983708876865 Max return 2222.958966954413
# ==================[Eval]====================
# Epi :  45
# Mean return  :  2112.1435030990747 Min return 1907.4318920735584 Max return 2314.980400795983
# ==================[Eval]====================
# Epi :  50
# Mean return  :  2491.0532034182143 Min return 2372.420364657995 Max return 2664.0018527408192
# ==================[Eval]====================
# Epi :  55
# Mean return  :  2777.585080494838 Min return 2622.0750776298864 Max return 2910.120005244983
# ==================[Eval]====================
# Epi :  60
# Mean return  :  2960.0128088661395 Min return 2605.5857519609635 Max return 3295.03377734764
# ==================[Eval]====================
# Epi :  65
# Mean return  :  3255.5474119933906 Min return 3182.6533000620734 Max return 3370.611296060349
# ==================[Eval]====================
# Epi :  70
# Mean return  :  3273.299893344458 Min return 2968.969787326988 Max return 3647.530164335298
# ==================[Eval]====================
# Epi :  75
# Mean return  :  3656.4925475471327 Min return 3335.456590347581 Max return 3953.2966867622417
# ==================[Eval]====================
# Epi :  80
# Mean return  :  2115.049516872768 Min return 121.22446436246135 Max return 3713.616817457097
# ==================[Eval]====================
# Epi :  85
# Mean return  :  3351.2849249811566 Min return 2166.9709224851917 Max return 3961.6815788404665
# ==================[Eval]====================
# Epi :  90
# Mean return  :  3956.08125605409 Min return 3315.1760968060744 Max return 4178.611314686324
# ==================[Eval]====================
# Epi :  95
# Mean return  :  3570.9902361220716 Min return 539.3976384762606 Max return 4382.387021663848
# ==================[Eval]====================
# Epi :  100
# Mean return  :  4457.134427236662 Min return 4323.730392071701 Max return 4616.6867927469475
# ==================[Eval]====================
# Epi :  105
# Mean return  :  4838.002149215142 Min return 4683.721762847343 Max return 5023.961297524786
# ==================[Eval]====================
# Epi :  110
# Mean return  :  5163.815221538047 Min return 5050.647829895319 Max return 5322.194483582558
# ==================[Eval]====================
# Epi :  115
# Mean return  :  5300.974583961386 Min return 5195.375250359442 Max return 5435.32817036284
# ==================[Eval]====================
# Epi :  120
# Mean return  :  5498.4820929241405 Min return 5344.857975960632 Max return 5631.551721652755
# ==================[Eval]====================
# Epi :  125
# Mean return  :  5105.527454569665 Min return 5048.123821165061 Max return 5166.924936819346
# ==================[Eval]====================
# Epi :  130
# Mean return  :  5157.664817387547 Min return 4911.970770496085 Max return 5326.393345688815
# ==================[Eval]====================
# Epi :  135
# Mean return  :  5440.781568034338 Min return 5205.721687335949 Max return 5691.414599314792
# ==================[Eval]====================
# Epi :  140
# Mean return  :  5917.838412508487 Min return 5734.148054276066 Max return 6064.441237174064
# ==================[Eval]====================
# Epi :  145
# Mean return  :  5266.611794078583 Min return 5027.063017887077 Max return 5394.852940294243
# ==================[Eval]====================
# Epi :  150
# Mean return  :  6127.534074350549 Min return 5903.5501203472695 Max return 6332.094381171218
# ==================[Eval]====================
# Epi :  155
# Mean return  :  5970.907518406652 Min return 5772.4954881712465 Max return 6154.990850532962
# ==================[Eval]====================
# Epi :  160
# Mean return  :  6157.321848108042 Min return 5992.308307327504 Max return 6289.878903832212
# ==================[Eval]====================
# Epi :  165
# Mean return  :  5877.585492380203 Min return 5622.2970998509 Max return 5994.08455183686
# ==================[Eval]====================
# Epi :  170
# Mean return  :  6138.298120194207 Min return 6016.515218222562 Max return 6228.9921867199355
# ==================[Eval]====================
# Epi :  175
# Mean return  :  6454.365344192329 Min return 6394.081401537085 Max return 6669.688346518031
# ==================[Eval]====================
# Epi :  180
# Mean return  :  6543.835040818452 Min return 6393.303870700406 Max return 6785.23333558759
# ==================[Eval]====================
# Epi :  185
# Mean return  :  6596.993461260766 Min return 6504.576264152742 Max return 6709.884307494542
# ==================[Eval]====================
# Epi :  190
# Mean return  :  6369.098357582408 Min return 6156.824641340039 Max return 6569.66396733842
# ==================[Eval]====================
# Epi :  195
# Mean return  :  6382.423420439054 Min return 6192.050151245623 Max return 6483.9053930329
# ==================[Eval]====================
# Epi :  200
# Mean return  :  6565.294715326789 Min return 6478.3996967205385 Max return 6650.936397718332
# ==================[Eval]====================
# Epi :  205
# Mean return  :  7093.0993990520565 Min return 6939.131146166947 Max return 7244.650297945905
# ==================[Eval]====================
# Epi :  210
# Mean return  :  6716.934824228435 Min return 6638.182941746941 Max return 6814.723097578214
# ==================[Eval]====================
# Epi :  215
# Mean return  :  6746.374660050458 Min return 6585.427770155962 Max return 6862.648843977207
# ==================[Eval]====================
# Epi :  220
# Mean return  :  7042.236524436892 Min return 6944.40018837708 Max return 7181.044439105782
# ==================[Eval]====================
# Epi :  225
# Mean return  :  7037.874151152995 Min return 6905.940096832123 Max return 7233.72332452967
# ==================[Eval]====================
# Epi :  230
# Mean return  :  6828.18303618909 Min return 6734.539848054385 Max return 6895.792414708386
# ==================[Eval]====================
# Epi :  235
# Mean return  :  7102.474516876971 Min return 7008.9528060535 Max return 7216.800976756106
# ==================[Eval]====================
# Epi :  240
# Mean return  :  7543.127857611013 Min return 7386.860609306517 Max return 7631.022626961584
# ==================[Eval]====================
# Epi :  245
# Mean return  :  7201.510197542992 Min return 7019.262668183376 Max return 7293.992554487884
# ==================[Eval]====================
# Epi :  250
# Mean return  :  7413.047305921839 Min return 7231.059760398607 Max return 7558.110884187169
# ==================[Eval]====================
# Epi :  255
# Mean return  :  7554.962995309213 Min return 7475.452782774992 Max return 7739.544204866408
# ==================[Eval]====================
# Epi :  260
# Mean return  :  7446.706238007743 Min return 7323.780525010848 Max return 7516.4882287764185
# ==================[Eval]====================
# Epi :  265
# Mean return  :  7260.583581012162 Min return 7140.952494526721 Max return 7445.7679223134655
# ==================[Eval]====================
# Epi :  270
# Mean return  :  7433.3495407271885 Min return 7237.213197601357 Max return 7681.5322829984725
# ==================[Eval]====================
# Epi :  275
# Mean return  :  7533.333415600986 Min return 7329.674655283763 Max return 7755.682170866068
# ==================[Eval]====================
# Epi :  280
# Mean return  :  8220.15285504443 Min return 8034.656111366347 Max return 8307.94388393037
# ==================[Eval]====================
# Epi :  285
# Mean return  :  7739.729996524747 Min return 7510.538477883217 Max return 8021.374201477051
# ==================[Eval]====================
# Epi :  290
# Mean return  :  7957.509494074557 Min return 7846.790089308469 Max return 8073.05348185761
# ==================[Eval]====================
# Epi :  295
# Mean return  :  7729.246669207796 Min return 7553.732394585153 Max return 7874.295001443306
# ==================[Eval]====================
# Epi :  300
# Mean return  :  8213.881408120662 Min return 7871.910093114169 Max return 8446.766162347954
# ==================[Eval]====================
# Epi :  305
# Mean return  :  8181.244490425946 Min return 8032.107002934343 Max return 8350.415096015353
# ==================[Eval]====================
# Epi :  310
# Mean return  :  8082.251738430151 Min return 7830.605741325228 Max return 8465.277512588642
# ==================[Eval]====================
# Epi :  315
# Mean return  :  8290.873254910352 Min return 8182.24035860539 Max return 8364.10557989249
# ==================[Eval]====================
# Epi :  320
# Mean return  :  8069.519827280834 Min return 7878.656700088865 Max return 8195.372094083592
# ==================[Eval]====================
# Epi :  325
# Mean return  :  7900.464940500555 Min return 7823.455590462711 Max return 8064.285708194391
# ==================[Eval]====================
# Epi :  330
# Mean return  :  8555.2539515117 Min return 8431.46241630314 Max return 8865.962310682173
# ==================[Eval]====================
# Epi :  335
# Mean return  :  8500.348365884103 Min return 8308.250807109931 Max return 8715.279706517564
# ==================[Eval]====================
# Epi :  340
# Mean return  :  8747.789778580918 Min return 8635.996432849954 Max return 8869.818245890268
# ==================[Eval]====================
# Epi :  345
# Mean return  :  8316.638425386484 Min return 8084.463307604407 Max return 8567.887258605499
# ==================[Eval]====================
# Epi :  350
# Mean return  :  7713.12523257109 Min return 7415.613946162509 Max return 8140.950997258558
# ==================[Eval]====================
# Epi :  355
# Mean return  :  8595.78409431673 Min return 8459.321128283094 Max return 8767.799616202408
# ==================[Eval]====================
# Epi :  360
# Mean return  :  8690.636337738095 Min return 8564.258979777367 Max return 8887.301292007412
# ==================[Eval]====================
# Epi :  365
# Mean return  :  8426.079200938424 Min return 8329.100124436858 Max return 8525.271577779149
# ==================[Eval]====================
# Epi :  370
# Mean return  :  8724.799250809803 Min return 8425.667809725957 Max return 9042.699177928618
# ==================[Eval]====================
# Epi :  375
# Mean return  :  9064.395704886181 Min return 8934.645462695968 Max return 9283.42305083096
# ==================[Eval]====================
# Epi :  380
# Mean return  :  8988.36071983318 Min return 8926.513862690928 Max return 9076.365192340922
# ==================[Eval]====================
# Epi :  385
# Mean return  :  8685.380602334819 Min return 8570.202004739805 Max return 8749.974285398763
# ==================[Eval]====================
# Epi :  390
# Mean return  :  8826.273990256725 Min return 8718.386987202399 Max return 8971.592377103703
# ==================[Eval]====================
# Epi :  395
# Mean return  :  9220.692722062557 Min return 8873.480542824178 Max return 9442.770279140745
# ==================[Eval]====================
# Epi :  400
# Mean return  :  8644.591457547698 Min return 8534.016688340938 Max return 8761.24410061874
# ==================[Eval]====================
# Epi :  405
# Mean return  :  9382.451531466846 Min return 9260.239232375327 Max return 9516.463337999132
# ==================[Eval]====================
# Epi :  410
# Mean return  :  9267.392504218773 Min return 9117.6877206039 Max return 9553.097923102347
# ==================[Eval]====================
# Epi :  415
# Mean return  :  9231.362031167886 Min return 9087.081699032944 Max return 9331.256476676996


