


from copy import deepcopy
import itertools
import numpy as np
from torch.optim import Adam
import gym
import time
from pytorch_shared import *
import torch
import torch.nn as nn
from common import *
from tensorboardX import SummaryWriter

import pointMass
from gym import wrappers
# @title SAC Model{ display-mode: "form" }
class SAC_model():

    def __init__(self, act_limit, obs_dim, act_dim, hidden_sizes, lr=1e-3, gamma=0.99, alpha=0.2, polyak=0.995,
                 load=False, exp_name='Exp1', replay_buffer = None, path='saved_models/'):
        self.act_limit = act_limit
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.load = load
        self.exp_name = exp_name
        self.path = path
        self.lr = lr
        self.create_networks(obs_dim, act_dim, hidden_sizes)
        self.replay_buffer = replay_buffer

        torch.set_num_threads(torch.get_num_threads())

    def create_networks(self, obs_dim, act_dim, hidden_sizes=[32, 32], batch_size=100, activation=nn.ReLU):

        # Get Env dimensions
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(self.act_limit, obs_dim, act_dim, hidden_sizes=hidden_sizes, activation=activation)
        self.actor  = self.ac # TODO COMPATABILITY
        if self.load:
            self.load_weights()
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _ = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
            # only need to do this because of PER
            #self.replay_buffer.update_priorities(data['PER_tree_idxs'], backup.cpu().numpy())

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        pi, logp_pi, _ = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def polyak_target_update(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def q_update(self,data):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

    def update(self,data):
        # First run one gradient descent step for Q1 and Q2
        self.q_update(data)
        # # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_target_update()

    def supervised_update(self, data, supervised_loss):
        # First run one gradient descent step for Q1 and Q2
        self.q_update(data)
        for p in self.q_params:
            p.requires_grad = False

        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi += supervised_loss
        loss_pi.backward()
        self.pi_optimizer.step()
        # Unfreeze Q-networks so you can optimize it at next step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_target_update()



    def load_weights(self, path='saved_models/'):
        self.ac.load_state_dict(torch.load(path+'_SAC_'+self.exp_name+'.pth'))
        self.ac.eval()
        print('Successfully loaded')

    def save_weights(self, path='saved_models/'):
        print('Saved at ', path+'_SAC_'+self.exp_name+'.pth')
        torch.save(self.ac.state_dict(), path+'_SAC_'+self.exp_name+'.pth')





def SAC(env_fn, ac_kwargs=dict(), seed=0,
        steps_per_epoch=2000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=200,
        max_ep_len=1000, save_freq=1, load=False, exp_name="Experiment_1", render=False):

    torch.manual_seed(seed) # TODO COMPAT
    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()


    test_env.reset()

    # test_env.render(mode='human')
    # env = wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    # test_env = wrappers.FlattenDictWrapper(test_env, dict_keys=['observation', 'desired_goal'])
    # Get Env dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    SAC = SAC_model(act_limit, obs_dim, act_dim, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load,
                    exp_name)

    # Logging
    start_time = time.time()
    train_log_dir = 'logs/' + exp_name + str(int(start_time))
    summary_writer = SummaryWriter(train_log_dir)

    def update_models(model, replay_buffer, steps, batch_size):
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size) #
            model.update(batch)

    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0

    if not load:
        # collect some initial random steps to initialise
        steps_collected += rollout_trajectories(n_steps=start_steps, env=env, max_ep_len=max_ep_len, actor='random',
                                                replay_buffer=replay_buffer, summary_writer=summary_writer,
                                                exp_name=exp_name)
        update_models(SAC, replay_buffer, steps=steps_collected, batch_size=batch_size)

    # now act with our actor, and alternately collect data, then train.
    while steps_collected < total_steps:
        # collect an episode
        steps_collected += rollout_trajectories(n_steps=max_ep_len, env=env, max_ep_len=max_ep_len,
                                                actor=SAC.actor.get_stochastic_action, replay_buffer=replay_buffer,
                                                summary_writer=summary_writer, current_total_steps=steps_collected,
                                                exp_name=exp_name)
        # take than many training steps
        update_models(SAC, replay_buffer, steps=max_ep_len, batch_size=batch_size)

        # if an epoch has elapsed, save and test.
        if steps_collected > 0 and steps_collected % steps_per_epoch == 0:
            SAC.save_weights()
            # Test the performance of the deterministic version of the agent.
            rollout_trajectories(n_steps=max_ep_len * 10, env=test_env, max_ep_len=max_ep_len,
                                 actor=SAC.actor.get_deterministic_action, summary_writer=summary_writer,
                                 current_total_steps=steps_collected, train=False, render=True, exp_name=exp_name)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    args = parser.parse_args()

    experiment_name = '_' + args.env + '_Hidden_' + str(args.hid) + 'l_' + str(args.l)

    SAC(lambda: gym.make(args.env),
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, load=args.load, exp_name=experiment_name,
        max_ep_len=args.max_ep_len, render=args.render)




