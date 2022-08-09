import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def weight_init_Xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped

class Qnet(nn.Module):
    def __init__(self, o_dim, a_dim, h_size=256):
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(o_dim + a_dim, h_size)
        self.fc2 = nn.Linear(h_size       , h_size)
        self.fc3 = nn.Linear(h_size       , 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    @multiple_action_q_function
    def forward(self,o_input:torch.Tensor,a_input:torch.Tensor):
        inputs = torch.concat([o_input,a_input],dim=-1)
        layer = self.relu1(self.fc1(inputs))
        layer = self.relu2(self.fc2(layer))
        qval  = self.fc3(layer)
        return torch.squeeze(qval,dim=-1)


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

class Policy(nn.Module):
    def __init__(self, o_dim, a_dim, h_size=256):
        super(Policy, self).__init__()
        self.LOG_SIG_MIN = -20.0
        self.LOG_SIG_MAX =   2.0
        self.a_dim = a_dim

        self.fc1 = nn.Linear(o_dim , h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc3_mu = nn.Linear(h_size, a_dim)
        self.fc3_log_sigma = nn.Linear(h_size, a_dim)
        self.Tanh = nn.Tanh()

    def forward(self,o_input, eval=False, repeat=None):
        if repeat is not None:
            o_input = extend_and_repeat(o_input, 1, repeat)
        layer = self.relu1(self.fc1(o_input))
        layer = self.relu2(self.fc2(layer))
        mu,log_sigma  = self.fc3_mu(layer), self.fc3_log_sigma(layer)
        sigma = torch.exp(torch.clip(log_sigma,self.LOG_SIG_MIN,self.LOG_SIG_MAX))
        dist  = Normal(mu,sigma)
        if eval:
            samples = mu
        else:
            samples = dist.rsample()

        actions = self.Tanh(samples)
        log_probs = dist.log_prob(samples).sum(axis=-1)

        log_probs -= torch.sum(torch.log(1 - actions**2 + 1e-10),axis=-1)
        return actions, log_probs


