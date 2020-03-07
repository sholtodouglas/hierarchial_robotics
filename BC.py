import numpy as np
from torch.optim import Adam
from pytorch_shared import *
import torch
import torch.nn as nn
import gym
import time
import pybullet
import reach2D
import os
import pointMass
from SAC import *
from common import *
from tensorboardX import SummaryWriter
from gym import wrappers



# Behavioural clone this mf.

def train_step(train_obs, train_acts, optimizer, policy, summary_writer, steps, batch_size = 512):
    indexes = np.random.choice(train_obs.shape[0], batch_size)
    obs, acts = train_obs[indexes, :], train_acts[indexes, :]
    optimizer.zero_grad()
    mu = policy(obs)
    loss = ((acts - mu) ** 2).mean()
    loss.backward()
    optimizer.step()
    summary_writer.add_scalar('BC_MSE_loss', loss, steps)
    return loss


def test_step(test_obs, test_acts, policy, summary_writer, steps, batch_size = 512):
    indexes = np.random.choice(test_obs.shape[0], batch_size)
    obs = test_obs[indexes, :]
    acts = test_acts[indexes, :]
    mu = policy(obs)
    loss = ((acts - mu) ** 2).mean()
    summary_writer.add_scalar('BC_MSE_loss', loss, steps)
    return loss


def save_policy(policy, exp_name, path = 'saved_models/'):
    try:
        os.makedirs(path + exp_name)
    except Exception as e:
        # print(e)
        pass
    torch.save(policy.state_dict(), path + exp_name + '/' + 'actor.pth')
    print('Saved model at: ', path + exp_name + '/' + 'actor.pth')

def behavioural_clone(filepath, env, exp_name, n_steps, batch_size, goal_based, architecture, max_ep_len = 200):
    # all data comes as [sequence, timesteps, dimension] so that when we are doing relay learning in the
    # trajectory we can't make mistakes about trajectory borders
    data = np.load(filepath)

    if goal_based:
        obs = np.concatenate([data['obs'], data['desired_goals']], axis=-1)
    else:
        obs = data['obs']

    acts = data['acts']

    obs, acts = torch.as_tensor(np.concatenate(obs, axis = 0), dtype=torch.float32).cuda(), torch.as_tensor(np.concatenate(acts, axis = 0), dtype=torch.float32).cuda()
    train_length = int(0.8 * (len(obs)))
    train_obs,train_acts  = obs[:train_length, :], acts[:train_length, :]
    valid_obs,  valid_acts  = obs[train_length:, :], acts[train_length:, :]
    print(train_obs.shape, valid_obs.shape)
    obs_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    act_dim, act_limit = env.action_space.shape[0], env.action_space.high[0]

    start_time = time.time()
    train_log_dir, valid_log_dir  = 'logs/' + 'BC_train_' +exp_name+'_:' + str(start_time),  'logs/' + 'BC_valid_' +exp_name+'_:' + str(start_time)
    train_summary_writer, valid_summary_writer = SummaryWriter(train_log_dir), SummaryWriter(valid_log_dir)

    policy = MLPActor(obs_dim, act_dim, hidden_sizes=architecture, act_limit=act_limit).cuda()
    optimizer = Adam(policy.parameters(), lr=1e-4)

    print('Done Initialisation, begin training')
    steps = 0
    while steps < n_steps:
        try:
            train_step(train_obs, train_acts, optimizer, policy, train_summary_writer, steps, batch_size)

            if steps % 500 == 0:
                l = test_step(valid_obs, valid_acts, policy, valid_summary_writer, steps, batch_size)
                print('Test Loss: ', steps, l)

            steps += 1

        except KeyboardInterrupt:
            txt = input("\nWhat would you like to do: ")
            if txt == 'v':
                print('Visualising')
                rollout_trajectories(n_steps = max_ep_len,env = env, max_ep_len = max_ep_len, actor = policy.get_deterministic_action, current_total_steps = steps, train = False, render = True, exp_name = exp_name, goal_based = True)
            print('Returning to Training.')
            if txt == 'q':
                raise Exception
            if txt == 's':
                save_policy(policy, exp_name)

    save_policy(policy, exp_name)






# python BC.py --filepath collected_data/1000HER2_pointMass-v0_Hidden_256l_2.npz
#  python BC.py --filepath collected_data/20000HER2_pointMassObject-v0_Hidden_256l_2.npz --env pointMassObject-v0


if __name__ == '__main__':
    import argparse
    print('hello')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default="")
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--goal_based', type=str2bool, default=True)


    args = parser.parse_args()
    if args.exp_name is None:
        exp_name = 'BC_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)
    else:
        exp_name = args.exp_name
    env = gym.make(args.env)
    behavioural_clone(args.filepath, env, exp_name, args.n_steps, args.batch_size, args.goal_based, [args.hid]*args.l)

