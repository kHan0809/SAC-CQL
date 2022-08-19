import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default="drawer-open-v2-goal-observable", help='box-close-v2-goal-observable drawer-open-v2-goal-observable drawer-close-v2-goal-observable')

    parser.add_argument('--device_eval',  default="cpu")
    parser.add_argument('--device_train', default="cuda")

    # ===================SAC hyperparameter======================
    parser.add_argument('--SAC_gamma', type=float, default=0.99) #TD3 공용
    parser.add_argument('--SAC_tau', type=float, default=0.005)  #TD3 공용
    parser.add_argument('--SAC_lr', type=float, default=3e-4)    #TD3 공용

    parser.add_argument('--SAC_hidden_size', type=int, default=256)  #TD3 공용
    parser.add_argument('--SAC_batch_size', type=int, default=128)   #TD3 공용
    parser.add_argument('--SAC_train_start', type=int, default=10000) #TD3 공용


    #====================TD3 hyperparameter======================
    parser.add_argument('--update_pi_ratio', type=int, default=2)

    #===================BCO hyperparameter=======================
    parser.add_argument('--BC_lr', type=float, default=3e-4)
    parser.add_argument('--BC_hidden_size', type=int, default=256)



    args = parser.parse_args()
    return args

