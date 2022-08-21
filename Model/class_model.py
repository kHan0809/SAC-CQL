import torch
import torch.nn as nn
from Model.model import Qnet, Policy, Det_Policy, soft_update
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import deque
from copy import deepcopy
import torch.nn.functional as F

class CustomDataSet(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class Buffer:
    def __init__(self,o_dim,a_dim,buffer_size = 1000000):
        self.size = buffer_size
        self.num_experience = 0
        self.o_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.a_mem = np.empty((self.size, a_dim), dtype=np.float32)
        self.no_mem = np.empty((self.size, o_dim), dtype=np.float32)
        self.r_mem = np.empty((self.size, 1), dtype=np.float32)
        self.done_mem = np.empty((self.size, 1), dtype=np.float32)
    def store_sample(self,o,a,r,no,done):
        idx = self.num_experience%self.size
        self.o_mem[idx] = o
        self.a_mem[idx] = a
        self.r_mem[idx] = r
        self.no_mem[idx] = no
        self.done_mem[idx] = done
        self.num_experience += 1
    def random_batch(self, batch_size = 256):
        N = min(self.num_experience, self.size)
        idx = np.random.choice(N,batch_size)
        o_batch = self.o_mem[idx]
        a_batch = self.a_mem[idx]
        r_batch = self.r_mem[idx]
        no_batch = self.no_mem[idx]
        done_batch = self.done_mem[idx]
        return o_batch, a_batch, r_batch, no_batch, done_batch
    def all_batch(self):
        N = min(self.num_experience,self.size)
        return self.o_mem[:N], self.a_mem[:N], self.r_mem[:N], self.no_mem[:N], self.done_mem[:N]
    def store_demo(self,demo):
        demo_len= len(demo)-1
        self.o_mem[:demo_len]  = demo[:-1]
        self.no_mem[:demo_len] = demo[1:]
        self.num_experience += demo_len



class SAC_Agent:
    def __init__(self,o_dim,a_dim,args):
        self.o_dim, self.a_dim = o_dim, a_dim
        self.args = args
        #SAC Hyperparameters
        self.gamma = args.SAC_gamma
        self.hidden_size = args.SAC_hidden_size
        self.batch_size = args.SAC_batch_size
        self.tau = args.SAC_tau
        self.training_start = args.SAC_train_start
        self.lr = args.SAC_lr

        #Define networks
        self.q1 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.q2 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.pi = Policy(self.o_dim, self.a_dim,self.hidden_size).to(args.device_train)
        self.buffer = Buffer(o_dim, a_dim)

        self.log_alpha = torch.tensor(0.0,requires_grad=True,device=args.device_train)
        #Define optimizer
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)


    def select_action(self,o,eval=False):
        action, _ = self.pi(torch.FloatTensor(o).to(self.args.device_train), eval)
        return action.cpu().detach().numpy()[0]

    def store_sample(self,o,a,r,no,done):
        self.buffer.store_sample(o,a,r,no,done)

    def train(self):
        if self.buffer.num_experience >= self.training_start:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.random_batch(self.args.SAC_batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
            action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
            reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
            done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)

            self.q_train(state_batch, action_batch,reward_batch,next_state_batch,done_batch)
            self.pi_train(state_batch)
            self.alpha_train(state_batch)
            self.target_q_update()

    def train_off(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)

        self.q_train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.pi_train(state_batch)
        self.alpha_train(state_batch)
        self.target_q_update()

    def q_train(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        with torch.no_grad():
            next_action_batch, next_log_pi = self.pi(next_state_batch)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*(minq - torch.exp(self.log_alpha)*next_log_pi)

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)

        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

    def pi_train(self,state_batch):
        self.pi_opt.zero_grad()
        action, log_pi = self.pi(state_batch)
        q_val1, q_val2 = self.q1(state_batch,action), self.q2(state_batch,action)
        pi_loss = (torch.exp(self.log_alpha)*log_pi - torch.min(q_val1,q_val2)).mean()
        pi_loss.backward()
        self.pi_opt.step()

    def alpha_train(self,state_batch):
        self.alpha_opt.zero_grad()
        _, log_pi = self.pi(state_batch)
        target_entropy = -self.a_dim
        alpha_loss = (torch.exp(self.log_alpha)*(-log_pi - target_entropy)).mean()
        alpha_loss.backward()
        self.alpha_opt.step()

    def target_q_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args.SAC_tau) + param.data * self.args.SAC_tau)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args.SAC_tau) + param.data * self.args.SAC_tau)

class SAC_CQL_Agent:
    def __init__(self,o_dim,a_dim,args):
        self.o_dim, self.a_dim = o_dim, a_dim
        self.args = args
        #SAC Hyperparameters
        self.gamma = args.SAC_gamma
        self.hidden_size = args.SAC_hidden_size
        self.batch_size = args.SAC_batch_size
        self.tau = args.SAC_tau
        self.lr = args.SAC_lr

        # CQL Hyperparmeters===나중에 args로 바꿔줘야함
        self.backup_entropy = False # 보류
        self.cql_n_actions = 10
        self.cql_importance_sample = True
        self.cql_lagrange = False
        self.cql_target_action_gap = 1.0
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.cql_max_target_backup = False
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max =  np.inf


        #Define networks
        self.q1 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.q2 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.pi = Policy(self.o_dim, self.a_dim,self.hidden_size).to(args.device_train)

        self.log_alpha = torch.tensor(0.0,requires_grad=True,device=args.device_train)

        #Define CQL value
        self.log_alpha_prime = torch.tensor(1.0, requires_grad=True, device=args.device_train)

        #Define optimizer
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha_prime_opt = torch.optim.Adam([self.log_alpha_prime], lr=self.lr)


    def select_action(self,o,eval=False):
        action, _ = self.pi(torch.FloatTensor(o).to(self.args.device_train), eval)
        return action.cpu().detach().numpy()[0]

    def store_sample(self,o,a,r,no,done):
        self.buffer.store_sample(o,a,r,no,done)

    def train(self,batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch


        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)


        pi_loss    = self.get_pi_loss(state_batch)
        q1_loss, q2_loss     = self.get_q_loss(state_batch, action_batch,reward_batch,next_state_batch,done_batch)
        alpha_loss = self.get_alpha_loss(state_batch)

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        self.q1_opt.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()


        self.target_q_update()


    def get_q_loss(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)

        next_action_batch, next_log_pi = self.pi(next_state_batch)
        next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
        minq = torch.min(next_q_val1,next_q_val2)
        if self.backup_entropy:
            target_ = reward_batch + self.gamma * (1 - done_batch) * (minq - torch.exp(self.log_alpha)*next_log_pi)
        else:
            target_ = reward_batch + self.gamma * (1 - done_batch) * (minq)

        q1_loss = F.mse_loss(target_.detach(),q_val1)
        q2_loss = F.mse_loss(target_.detach(),q_val2)


        #====여까지는 그냥 SAC랑 같음
        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        cql_random_actions = action_batch.new_empty((batch_size, self.cql_n_actions, action_dim),requires_grad=False).uniform_(-1, 1)


        cql_current_actions, cql_current_log_pis = self.pi(state_batch, repeat=self.cql_n_actions)
        cql_next_actions, cql_next_log_pis = self.pi(next_state_batch, repeat=self.cql_n_actions)
        cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()


        cql_q1_rand = self.q1(state_batch, cql_random_actions)
        cql_q2_rand = self.q2(state_batch, cql_random_actions)
        cql_q1_current_actions = self.q1(state_batch, cql_current_actions)
        cql_q2_current_actions = self.q2(state_batch, cql_current_actions)
        cql_q1_next_actions = self.q1(state_batch, cql_next_actions)
        cql_q2_next_actions = self.q2(state_batch , cql_next_actions)

        cql_cat_q1 = torch.cat(
            [cql_q1_rand, torch.unsqueeze(q_val1, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
        )
        cql_cat_q2 = torch.cat(
            [cql_q2_rand, torch.unsqueeze(q_val2, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
        )
        # cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        # cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat(
                [cql_q1_rand - random_density,
                 cql_q1_next_actions - cql_next_log_pis.detach(),
                 cql_q1_current_actions - cql_current_log_pis.detach()],
                dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand - random_density,
                 cql_q2_next_actions - cql_next_log_pis.detach(),
                 cql_q2_current_actions - cql_current_log_pis.detach()],
                dim=1
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_val1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_val2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()


        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime), min=0.0, max=1000000.0)
            cql_min_qf1_loss = alpha_prime * self.cql_min_q_weight * (
                        cql_qf1_diff - self.cql_target_action_gap)
            cql_min_qf2_loss = alpha_prime * self.cql_min_q_weight * (
                        cql_qf2_diff - self.cql_target_action_gap)

            self.alpha_prime_opt.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_opt.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        q1_loss_add_cql =  q1_loss + cql_min_qf1_loss
        q2_loss_add_cql =  q2_loss + cql_min_qf2_loss

        return q1_loss_add_cql, q2_loss_add_cql


    def get_pi_loss(self,state_batch):
        action, log_pi = self.pi(state_batch)
        q_val1, q_val2 = self.q1(state_batch,action), self.q2(state_batch,action)
        pi_loss = (torch.exp(self.log_alpha)*log_pi - torch.min(q_val1,q_val2)).mean()
        return pi_loss

    def get_alpha_loss(self,state_batch):
        _, log_pi = self.pi(state_batch)
        target_entropy = -self.a_dim
        alpha_loss = (torch.exp(self.log_alpha)*(-log_pi - target_entropy).detach()).mean()
        return  alpha_loss


    def target_q_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args.SAC_tau) + param.data * self.args.SAC_tau)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args.SAC_tau) + param.data * self.args.SAC_tau)



class TD3_Agent:
    def __init__(self,o_dim,a_dim,args):
        self.o_dim, self.a_dim = o_dim, a_dim
        self.args = args
        #SAC Hyperparameters
        self.gamma = args.SAC_gamma
        self.hidden_size = args.SAC_hidden_size
        self.batch_size = args.SAC_batch_size
        self.tau = args.SAC_tau
        self.training_start = args.SAC_train_start
        self.lr = args.SAC_lr
        self.update_pi = args.update_pi_ratio
        self.update_count = 0

        #Define networks
        self.q1 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.q2 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        self.pi = Det_Policy(self.o_dim, self.a_dim,self.hidden_size).to(args.device_train)
        self.target_pi = deepcopy(self.pi)
        self.buffer = Buffer(o_dim, a_dim)

        #Define optimizer
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=self.lr)
        #====cql hyper====
        # CQL Hyperparmeters===나중에 args로 바꿔줘야함
        self.backup_entropy = False # 보류
        self.cql_n_actions = 10
        self.cql_importance_sample = True
        self.cql_lagrange = False
        self.cql_target_action_gap = 1.0
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.cql_max_target_backup = False
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max =  np.inf

    def init_pi(self,dir):
        self.pi.load_state_dict(torch.load(dir)['policy'])
        self.target_pi = deepcopy(self.pi)
    def init_q(self,dir):
        self.q1.load_state_dict(torch.load(dir)['q1'])
        self.q2.load_state_dict(torch.load(dir)['q2'])
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

    def select_action(self,o,eval=False):
        o = o.reshape([1,-1])
        if eval:
            action = self.pi(torch.FloatTensor(o).to(self.args.device_train))
            return  action.cpu().detach().numpy()[0]
        else:
            action = (self.pi(torch.FloatTensor(o).to(self.args.device_train)).cpu().detach().numpy()[0] + \
                     + 0.1 * np.random.normal(0.0, 1.0, [self.a_dim])).clip(-1.0,1.0)
            return action

    def store_sample(self,o,a,r,no,done):
        self.buffer.store_sample(o,a,r,no,done)

    def train(self):
        if self.buffer.num_experience >= self.training_start:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.random_batch(self.args.SAC_batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
            action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
            reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train).reshape(state_batch.shape[0])
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
            done_batch = torch.FloatTensor(done_batch).to(self.args.device_train).reshape(state_batch.shape[0])

            self.q_train(state_batch, action_batch,reward_batch,next_state_batch,done_batch)

            if (self.update_count%self.update_pi) == 0:
                self.pi_train(state_batch)
                with torch.no_grad():
                    soft_update(self.target_q1, self.q1, self.tau)
                    soft_update(self.target_q2, self.q2, self.tau)
                    soft_update(self.target_pi, self.pi, self.tau)

            self.update_count += 1

    def train_off(self, batch, cql=False):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)
        if cql:
            self.q_train_cql(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        else:
            self.q_train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if (self.update_count % self.update_pi) == 0:
            self.pi_train(state_batch)
            # with torch.no_grad():
            #     soft_update(self.target_q1, self.q1, self.tau)
            #     soft_update(self.target_q2, self.q2, self.tau)
            #     soft_update(self.target_pi, self.pi, self.tau)

        if (self.update_count % 2.0) == 0:
            with torch.no_grad():
                soft_update(self.target_q1, self.q1, self.tau)
                soft_update(self.target_q2, self.q2, self.tau)
                soft_update(self.target_pi, self.pi, self.tau)

        self.update_count += 1

    def train_Only_Q(self, batch,cql=False):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)

        if cql:
            self.q_train_cql(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        else:
            self.q_train(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        with torch.no_grad():
            soft_update(self.target_q1, self.q1, self.tau)
            soft_update(self.target_q2, self.q2, self.tau)

    def test_q(self,batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        with torch.no_grad():
            q_val1, q_val2 = self.q1(state_batch, action_batch), self.q2(state_batch, action_batch)
        return q_val1, q_val2

    def q_train(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        # reward_batch, done_batch = reward_batch.reshape(q_val1.shape[0]), done_batch.reshape(q_val1.shape[0])
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action_batch = (self.target_pi(next_state_batch) + noise).clamp(-1.,1.)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*minq

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)

        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

    def q_train_cql(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action_batch = (self.target_pi(next_state_batch) + noise).clamp(-1.,1.)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*minq

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)


        #====여까지는 그냥 SAC랑 같음
        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        cql_random_actions = action_batch.new_empty((batch_size, self.cql_n_actions, action_dim),requires_grad=False).uniform_(-1, 1)


        cql_current_actions = self.pi(state_batch, repeat=self.cql_n_actions)
        cql_next_actions    = self.pi(next_state_batch, repeat=self.cql_n_actions)
        cql_current_actions, cql_next_actions = cql_current_actions.detach(), cql_next_actions.detach()



        cql_q1_rand = self.q1(state_batch, cql_random_actions)
        cql_q2_rand = self.q2(state_batch, cql_random_actions)
        cql_q1_current_actions = self.q1(state_batch, cql_current_actions)
        cql_q2_current_actions = self.q2(state_batch, cql_current_actions)
        cql_q1_next_actions = self.q1(state_batch, cql_next_actions)
        cql_q2_next_actions = self.q2(state_batch , cql_next_actions)

        cql_cat_q1 = torch.cat(
            [cql_q1_rand, torch.unsqueeze(q_val1, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
        )
        cql_cat_q2 = torch.cat(
            [cql_q2_rand, torch.unsqueeze(q_val2, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
        )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_val1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_val2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()


        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime), min=0.0, max=1000000.0)
            cql_min_qf1_loss = alpha_prime * self.cql_min_q_weight * (
                        cql_qf1_diff - self.cql_target_action_gap)
            cql_min_qf2_loss = alpha_prime * self.cql_min_q_weight * (
                        cql_qf2_diff - self.cql_target_action_gap)

            self.alpha_prime_opt.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_opt.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        q1_loss_add_cql =  q1_loss + cql_min_qf1_loss
        q2_loss_add_cql =  q2_loss + cql_min_qf2_loss


        q1_loss_add_cql.backward(retain_graph=True)
        self.q1_opt.step()
        q2_loss_add_cql.backward()
        self.q2_opt.step()


    def pi_train(self,state_batch):
        self.pi_opt.zero_grad()
        action = self.pi(state_batch)
        pi_loss = -self.q1(state_batch,action).mean()
        pi_loss.backward()
        self.pi_opt.step()




class BC_agent:
    def __init__(self,o_dim,a_dim,args):
        self.o_dim, self.a_dim = o_dim, a_dim
        self.args = args
        self.lr = args.BC_lr
        self.bc = Det_Policy(o_dim,a_dim,args.BC_hidden_size).to(args.device_train)
        self.bc_opt = torch.optim.Adam(self.bc.parameters(), lr=self.lr)

        self.hidden_size = args.SAC_hidden_size
        #Define networks
        self.q1 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.q2 = Qnet(self.o_dim, self.a_dim, self.hidden_size).to(args.device_train)
        self.n_actions = 50
        self.update_count = 0
        self.update_pi = args.update_pi_ratio

    def init_bc(self,dir):
        self.bc.load_state_dict(torch.load(dir)['policy'])
    def init_q(self,dir):
        self.q1.load_state_dict(torch.load(dir)['q1'])
        self.q2.load_state_dict(torch.load(dir)['q2'])
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

    def select_action(self, o, eval=False):
        action  = self.bc(torch.FloatTensor(o).to(self.args.device_train))
        return action.cpu().detach().numpy()[0]

    def train(self, batch):
        self.bc.train()
        # if self.buffer.num_experience >= self.training_start:
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.random_batch(self.args.SAC_batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)

        self.bc_opt.zero_grad()
        pred_action = self.bc(state_batch)
        action_loss = F.mse_loss(pred_action,action_batch)
        action_loss.backward()
        self.bc_opt.step()

    def train_weightedQ(self,batch,cql=False):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
        action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
        reward_batch = torch.FloatTensor(reward_batch).to(self.args.device_train)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.args.device_train)
        done_batch = torch.FloatTensor(done_batch).to(self.args.device_train)


        self.train_bc(state_batch,action_batch,reward_batch,next_state_batch,done_batch)

        if cql:
            self.q_train_cql(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        else:
            self.q_train_cql(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if (self.update_count % self.update_pi) == 0:
            self.pi_train(state_batch)

        if (self.update_count % 2.0) == 0:
            with torch.no_grad():
                soft_update(self.target_q1, self.q1, self.tau)
                soft_update(self.target_q2, self.q2, self.tau)

        self.update_count += 1



    def train_bc(self,state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.bc.train()
        # 방법1 uniform norm. 방법2 batch norm
        random_actions = action_batch.new_empty((state_batch.shape[0], self.n_actions, self.a_dim),requires_grad=False).uniform_(-1, 1)
        with torch.no_grad():
            q_random_val1,q_random_val2 = self.q1(state_batch,random_actions), self.q2(state_batch,random_actions) #state는 그대로 [256,17] action은 [256,10,6이 들어가면된다.]
            min_q = torch.min(q_random_val1, q_random_val2).mean(dim=1)

            q_val1, q_val2 = self.q1(state_batch, action_batch), self.q2(state_batch, action_batch)

            weight = torch.min((q_val1-min_q)/abs(min_q),(q_val2-min_q)/abs(min_q)).clamp(0.0,2.0)

        self.bc_opt.zero_grad()
        pred_action = self.bc(state_batch)
        action_loss = torch.mean(((pred_action - action_batch)**2)*weight.reshape(-1, 1))
        action_loss.backward()
        self.bc_opt.step()


    def q_train(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        # reward_batch, done_batch = reward_batch.reshape(q_val1.shape[0]), done_batch.reshape(q_val1.shape[0])
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action_batch = (self.bc(next_state_batch) + noise).clamp(-1.,1.)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*minq

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)

        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

    def q_train_cql(self,state_batch,action_batch,reward_batch,next_state_batch,done_batch):
        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_val1,q_val2 = self.q1(state_batch,action_batch), self.q2(state_batch,action_batch)
        with torch.no_grad():
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action_batch = (self.bc(next_state_batch) + noise).clamp(-1.,1.)
            next_q_val1, next_q_val2 = self.target_q1(next_state_batch,next_action_batch), self.target_q2(next_state_batch,next_action_batch)
            minq = torch.min(next_q_val1,next_q_val2)
            target_ = reward_batch + self.gamma*(1-done_batch)*minq

        q1_loss = F.mse_loss(target_,q_val1)
        q2_loss = F.mse_loss(target_,q_val2)


        #====여까지는 그냥 SAC랑 같음
        batch_size = action_batch.shape[0]
        action_dim = action_batch.shape[-1]
        cql_random_actions = action_batch.new_empty((batch_size, self.cql_n_actions, action_dim),requires_grad=False).uniform_(-1, 1)


        cql_current_actions = self.bc(state_batch, repeat=self.cql_n_actions)
        cql_next_actions    = self.bc(next_state_batch, repeat=self.cql_n_actions)
        cql_current_actions, cql_next_actions = cql_current_actions.detach(), cql_next_actions.detach()



        cql_q1_rand = self.q1(state_batch, cql_random_actions)
        cql_q2_rand = self.q2(state_batch, cql_random_actions)
        cql_q1_current_actions = self.q1(state_batch, cql_current_actions)
        cql_q2_current_actions = self.q2(state_batch, cql_current_actions)
        cql_q1_next_actions = self.q1(state_batch, cql_next_actions)
        cql_q2_next_actions = self.q2(state_batch , cql_next_actions)

        cql_cat_q1 = torch.cat(
            [cql_q1_rand, torch.unsqueeze(q_val1, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1
        )
        cql_cat_q2 = torch.cat(
            [cql_q2_rand, torch.unsqueeze(q_val2, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1
        )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q_val1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q_val2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()


        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime), min=0.0, max=1000000.0)
            cql_min_qf1_loss = alpha_prime * self.cql_min_q_weight * (
                        cql_qf1_diff - self.cql_target_action_gap)
            cql_min_qf2_loss = alpha_prime * self.cql_min_q_weight * (
                        cql_qf2_diff - self.cql_target_action_gap)

            self.alpha_prime_opt.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_opt.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

        q1_loss_add_cql =  q1_loss + cql_min_qf1_loss
        q2_loss_add_cql =  q2_loss + cql_min_qf2_loss


        q1_loss_add_cql.backward(retain_graph=True)
        self.q1_opt.step()
        q2_loss_add_cql.backward()
        self.q2_opt.step()



    #
    #
    # def train_weightedQ(self,batch):
    #     self.bc.train()
    #     state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    #     state_batch = torch.FloatTensor(state_batch).to(self.args.device_train)
    #     action_batch = torch.FloatTensor(action_batch).to(self.args.device_train)
    #     # 방법1 uniform norm. 방법2 batch norm
    #     random_actions = action_batch.new_empty((state_batch.shape[0], self.n_actions, self.a_dim),requires_grad=False).uniform_(-1, 1)
    #     with torch.no_grad():
    #         q_random_val1,q_random_val2 = self.q1(state_batch,random_actions), self.q2(state_batch,random_actions) #state는 그대로 [256,17] action은 [256,10,6이 들어가면된다.]
    #         min_q = torch.min(q_random_val1, q_random_val2).mean(dim=1)
    #
    #         q_val1, q_val2 = self.q1(state_batch, action_batch), self.q2(state_batch, action_batch)
    #
    #         weight = torch.min((q_val1-min_q)/abs(min_q),(q_val2-min_q)/abs(min_q)).clamp(0.0,2.0)
    #
    #     self.bc_opt.zero_grad()
    #     pred_action = self.bc(state_batch)
    #     action_loss = torch.mean(((pred_action - action_batch)**2)*weight.reshape(-1, 1))
    #     action_loss.backward()
    #     self.bc_opt.step()



