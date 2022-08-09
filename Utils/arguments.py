import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default="drawer-open-v2-goal-observable", help='box-close-v2-goal-observable drawer-open-v2-goal-observable drawer-close-v2-goal-observable')

    parser.add_argument('--device_eval',  default="cpu")
    parser.add_argument('--device_train', default="cuda")

    # ===================SAC hyperparameter======================
    parser.add_argument('--SAC_gamma', type=float, default=0.99)
    parser.add_argument('--SAC_tau', type=float, default=0.005)
    parser.add_argument('--SAC_lr', type=float, default=3e-4)

    parser.add_argument('--SAC_hidden_size', type=int, default=256)
    parser.add_argument('--SAC_batch_size', type=int, default=128)
    parser.add_argument('--SAC_train_start', type=int, default=1000)



    #===================BCO hyperparameter======================
    parser.add_argument('--BCO_T', type=int, default=2000)
    parser.add_argument('--BCO_epi_len', type=int, default=500)
    parser.add_argument('--BCO_state_dim', type=int, default=25)
    parser.add_argument('--BCO_InvKin_epoch', type=int, default=20)
    parser.add_argument('--BCO_lr', type=float, default=1e-4)
    parser.add_argument('--BCO_wd', type=float, default=1e-5)

    parser.add_argument('--BCO_BC_epoch', type=int, default=20)
    parser.add_argument('--BCO_eval_epi', type=int, default=5)

    #===================IGL SAC hyperparameter====================
    parser.add_argument('--IGLSAC_ONLY_IGL', type=int, default=1000)


    args = parser.parse_args()
    return args

