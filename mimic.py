
import gym
import pybullet
import pointMass #  the act of importing registers the env.
import pandaRL
from common import *
from SAC import *
from TD3 import *
import numpy as np
from pytorch_shared import *
import torch
from common import *
from tensorboardX import SummaryWriter
from datetime import datetime
from BC import find_supervised_loss, load_data
import time

def mimic_rollouts(n_steps, env, trajectory, max_ep_len=200,actor=None, replay_buffer=None, summary_writer=None,
                         current_total_steps=0,
                         render=False, train=True, exp_name=None, s_g=None, return_episode=False,
                         replay_trajectory=None, start_state=None, goal_based=False, lstm_actor=None,
                         replay_obs=None, extra_info=None, model = None, batch_size=None, supervised_kwargs = None, supervised_func = None):


    ###################  quick fix for the need for this to activate rendering pre env reset.  ###################
    ###################  MUST BE A BETTER WAY? Env realising it needs to change pybullet client?  ###################
    if 'reacher' in exp_name or 'point' in exp_name or 'robot' in exp_name or 'UR5' in exp_name:
        pybullet = True
    else:
        pybullet = False

    if pybullet:
        if render:
            # have to do it beforehand to connect up the pybullet GUI
            env.render(mode='human')

    ###################  ###################  ###################  ###################  ###################

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    if start_state is not None:
        o = env.reset(start_state)
    if s_g is not None:
        env.reset_goal_pos(s_g)

    # if we want to store expert actions

    if return_episode:
        episode_buffer = []
        episode = []

    if lstm_actor is not None:
        past_state = [None]

    for t in range(n_steps):
        if actor == 'random':
            a = env.action_space.sample()
        elif replay_trajectory is not None:  # replay a trajectory that we've fed in so that we can make sure this is properly deterministic and compare it to our estimated action based trajectory/

            a = replay_trajectory[t]
        elif replay_obs is not None:

            a, past_state = lstm_actor(np.concatenate([replay_obs[t], o['desired_goal']], axis=0),
                                       past_state=past_state)
        elif lstm_actor is not None:
                # print(o['observation'],z,o['desired_goal'])
                a_base, past_state = lstm_actor(np.concatenate([o['observation'], o['desired_goal']], axis=0),
                                                past_state=past_state)
        elif goal_based:
            a = actor(np.concatenate([o['observation'], o['desired_goal']], axis=0))
        else:
            a = actor(o)
        # Step the env
        o2, r, d, _ = env.step(a)

        # mimicry reward
        r = -np.linalg.norm(o2['full_positional_state'] - trajectory[t])

        if render:
            env.render(mode='human')
            env.visualise_sub_goal(trajectory[t], sub_goal_state = 'full_positional_state')

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer # dont use a replay buffer with HER, because we need to take entire episodes and do some computation so that happens after.
        if replay_buffer:
            o_store = np.concatenate([o['observation'], o['desired_goal']])
            o2_store = np.concatenate([o2['observation'], o2['desired_goal']])
            replay_buffer.store(o_store, a, r, o2_store, d)
            batch = replay_buffer.sample_batch(batch_size)
            if supervised_func:
                # if we have a function which provides a supervised loss term, use it.
                supervised_kwargs['steps'] = current_total_steps + t
                BC_loss = supervised_func(**supervised_kwargs)
                model.supervised_update(batch, BC_loss)
            else:
                model.update(batch)

        if return_episode:
                episode.append([o, a, r, o2, d])  # add the full transition to the episode.


        # Super critical, easy to overlook step: make sure to update
        # most recent observation!

        o = o2
        # if either we've ended an episdoe, collected all the steps or have reached max ep len and
        # thus need to log ep reward and reset
        if d or (ep_len == int(max_ep_len)) or (t == int((n_steps - 1))):
            if return_episode:
                episode_buffer.append(episode)
                episode = []
            if train:
                print('Frame: ', t + current_total_steps, ' Return: ', ep_ret)
                summary_string = 'Episode_return'
            else:
                print('Test Frame: ', t + current_total_steps, ' Return: ', ep_ret)
                summary_string = 'Test_Episode_return'

            if summary_writer:
                summary_writer.add_scalar(summary_string, ep_ret, t + current_total_steps)

            # reset the env if there are still steps to collect
            if t < n_steps - 1:
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # else it will be the end of the loop and of the function.
    if return_episode:
        return {'episodes': episode_buffer, 'n_steps': n_steps}
    return n_steps




def select_trajectory(data):
    # data comes in num eps, length, dim for each thing.
    idx = np.random.choice(len(data['acts']))
    start = np.random.choice(int(len(data['acts'][idx])*0.2))
    length = len(data['acts'][idx])
    end = np.random.randint( int(length*0.8), length)
    trajectory_ags = data['full_positional_states'][idx, start:end, :]
    init = data['obs'][idx, start, :]
    goal = data['achieved_goals'][idx, end, :]
    acts = data['acts'][idx, start:end, :]
    return init, goal, trajectory_ags, acts


# python mimic.py --env pointMass-v0 --BC_filepath collected_data/10000experiment_2_HER_pointMass-v0.npz

def training_loop(env_fn, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=0,
                  max_ep_len=300, load=False, exp_name="Experiment_1", render=False,
                  BC_filepath=None, play=False):

    torch.manual_seed(seed)
    np.random.seed(seed)


    test_env = env_fn()
    env = env_fn()

    # pybullet needs the GUI env to be reset first for our noncollision stuff to work.
    if render:
        test_env.render(mode='human')
        test_env.reset()

    # Get Env dimensions
    obs_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    model = SAC_model(act_limit, obs_dim, act_dim, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load,
                    exp_name)

    # Logging
    train_log_dir = 'logs/' + exp_name+str(datetime.now())
    summary_writer = SummaryWriter(train_log_dir)

    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0
    epoch_ticker = 0

    rollout_rl_kwargs = {'n_steps': max_ep_len, 'env': env, 'max_ep_len': max_ep_len,
                         'actor': model.actor.get_stochastic_action,
                         'summary_writer': summary_writer, 'exp_name': exp_name, 'return_episode': True,
                         'goal_based': True,
                         'replay_buffer': replay_buffer, 'model': model, 'batch_size': batch_size,
                         'current_total_steps': steps_collected
        , 'supervised_kwargs': None, 'supervised_func': None}

    # if BC_filepath:
    #     train_obs, train_acts, valid_obs, valid_acts = load_data(BC_filepath, goal_based=True)
    #     BC_kwargs = {'obs': train_obs, 'acts': train_acts, 'optimizer': model.pi_optimizer, 'policy': model.ac.pi,
    #                  'summary_writer': summary_writer, 'steps': steps_collected}
    #     rollout_rl_kwargs['supervised_kwargs'], rollout_rl_kwargs['supervised_func'] = BC_kwargs, find_supervised_loss

    data = np.load(BC_filepath)
    rollout_random_kwargs = rollout_rl_kwargs.copy()
    rollout_random_kwargs['actor'] = 'random'
    rollout_random_kwargs['n_steps'] = start_steps

    rollout_viz_kwargs = rollout_rl_kwargs.copy()
    rollout_viz_kwargs['env'] = test_env
    rollout_viz_kwargs['train'] = False
    rollout_viz_kwargs['render'] = render
    rollout_viz_kwargs['replay_buffer'] = None
    rollout_viz_kwargs['actor'] = model.actor.get_deterministic_action

    def rollout(rollout_kwargs, steps_collected, epoch_ticker, data):
        start, end, trajectory_full_pos, acts = select_trajectory(data)
        rollout_kwargs['start_state'] = start
        rollout_kwargs['s_g'] = end
        rollout_kwargs['trajectory'] = trajectory_full_pos
        #rollout_kwargs['replay_trajectory'] = acts # activate for reenactment of example acts
        rollout_kwargs['n_steps'] = len(acts)
        #todo - check if it works off a dataset of densy bois. Do per timestep reward. 
        episodes = mimic_rollouts(**rollout_kwargs)
        steps_collected += episodes['n_steps']
        if steps_collected >= epoch_ticker:
            model.save_weights()
            epoch_ticker += steps_per_epoch
        return steps_collected, epoch_ticker


    if play:
        while (1):
            rollout_viz_kwargs['n_steps'] = max_ep_len
            rollout_viz_kwargs['current_total_steps'] += 1
            steps_collected, epoch_ticker = rollout(rollout_viz_kwargs, steps_collected, epoch_ticker, data)

    while steps_collected < total_steps:
        try:
            rollout_rl_kwargs['current_total_steps'] = steps_collected
            steps_collected, epoch_ticker = rollout(rollout_rl_kwargs, steps_collected, epoch_ticker, data)

        except KeyboardInterrupt:
            txt = input("\nWhat would you like to do: ")
            if txt.isnumeric():
                for i in range(0,int(txt)):
                    rollout_rl_kwargs['current_total_steps'] = steps_collected
                    steps_collected, epoch_ticker = rollout(rollout_viz_kwargs, steps_collected, epoch_ticker, data)
            print('Returning to Training.')
            if txt == 'q':
                raise Exception


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--max_ep_len', type=int,
                        default=150)  # fetch reach learns amazingly if 50, but not if 200 Why? Because thats the interval we add hindsight episodes at!
    parser.add_argument('--exp_name', type=str, default='experiment_2')
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--BC_filepath', type=str, default=None)
    parser.add_argument('--play', type=str2bool, default=False)

    args = parser.parse_args()
    exp_name = args.exp_name + '_HER_' + args.env
    save_file(__file__, exp_name, args)

    training_loop(lambda: gym.make(args.env),
                  ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  gamma=args.gamma, seed=args.seed, epochs=args.epochs, load=args.load, exp_name=exp_name,
                  max_ep_len=args.max_ep_len, render=True, BC_filepath=args.BC_filepath
                  , play=args.play)
