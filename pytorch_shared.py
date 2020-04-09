

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from PrioritizedReplayBuffer import PER

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit, activation=nn.ReLU):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

    def get_deterministic_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).cuda()
        return self.forward(obs).cpu().detach().numpy()

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes,  act_limit, activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):

        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = torch.tensor(self.act_limit).cuda() * pi_action

        return pi_action, logp_pi, pi_distribution


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCriticTD3(nn.Module):

    def __init__(self, act_limit, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU, noise_scale=0.1):
        super().__init__()

        # build policy and value functions
        self.noise_scale = noise_scale
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, act_limit, activation).cuda()
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).cuda()
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).cuda()

    def act(self, obs, noise=0):
        obs = torch.as_tensor(obs, dtype=torch.float32).cuda()
        with torch.no_grad():
            a  = self.pi(obs).cpu().numpy()
            a += noise * np.random.randn(self.act_dim)
            return np.clip(a, -self.act_limit, self.act_limit)

    # convenience functions for passing function as a param
    def get_deterministic_action(self, obs):
        return self.act(obs, noise = 0)

    def get_stochastic_action(self, obs):
        return self.act(obs, noise = self.noise_scale)


class MLPActorCritic(nn.Module):

    def __init__(self, act_limit, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        # build policy and value functions

        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, act_limit, activation).cuda()
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).cuda()
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).cuda()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ , _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()

    # convenience functions for passing function as a param
    def get_deterministic_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).cuda()
        return self.act(obs, deterministic = True)

    def get_stochastic_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32).cuda()
        return self.act(obs, deterministic = False)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros( combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros( combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros( combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.PER = PER(size)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.PER.store(self.ptr) # PER store index
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):

        tree_idxs, idxs = self.PER.sample(batch_size)
        #idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        batch =  {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k, v in batch.items()}
        batch['PER_tree_idxs'] = tree_idxs
        return batch

    def update_priorities(self, tree_idxs, TD_errors):
        self.PER.batch_update(tree_idxs, TD_errors)