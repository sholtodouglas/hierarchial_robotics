
import numpy as np
import tensorflow as tf
import gym
import pybullet
import pointMass #  the act of importing registers the env.
import ur5_RL
import time
from common import *
from SAC import *
from TD3 import *
from HER import *
import copy
import psutil
import multiprocessing as mp
from tqdm import tqdm
from natsort import natsorted, ns

# lower achieved_whole_state is whether to use the full state
# substitute action is whether to swap out the action commanded by the higher level network for the actual achieved state of the lower level one. Following the openai RFR and Levy's paper.
# use_higher_level is whether to actually use the higher level net or just use the lower one.  (use this mostly for testing).

def rollout_trajectories_hierarchially(n_steps, env, max_ep_len=200, actor_lower=None, actor_higher= None, replan_interval = 30, lower_achieved_whole_state = True, substitute_action = True, use_higher_level = True, summary_writer=None,
                         current_total_steps=0,
                         render=False, train=True, exp_name=None, s_g=None, return_episode=False,
                         replay_trajectory=None,
                         compare_states=None, start_state=None, lstm_actor=None,
                         only_use_baseline=False,
                         replay_obs=None, extra_info=None):


    # reset the environment
    def set_init(o, env, extra_info):

        if extra_info is not None:

            env.initialize_start_pos(start_state, extra_info)
        else:
            env.initialize_start_pos(start_state)  # init vel to 0, but x and y to the desired pos.
        o['observation'] = start_state


        return o

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
        o = set_init(o, env, extra_info)
    if s_g is not None:
        env.reset_goal_pos(s_g)

    # if we want to store expert actions

    if return_episode:
        episode_buffer_lower = []
        episode_buffer_higher = []
        episode_lower = []
        episode_higher = []

    if lstm_actor is not None:
        past_state = [None]

    higher_level_steps = 0
    for t in range(n_steps):

        if t % replan_interval == 0:
            if t > 0: # this is the second time we have entered this, so we can begin storing transitions
                if substitute_action:
                    if lower_achieved_whole_state:
                        action = o2['full_positional_state']
                    else:
                        action = o2['controllable_achieved_goal']
                     # validate actions as though the lower level is actually good at achieving goals.
                else:
                    action = sub_goal  # subgoal will already be defined.
                if return_episode:
                    episode_higher.append([higher_o1, action,r,o2, d])
                higher_level_steps += 1


            higher_o1 = o  # keep it here for storage for the next transition
            if use_higher_level:
                sub_goal = actor_higher(np.concatenate([o['observation'], o['desired_goal']], axis=0))
            else:

                sub_goal = o['desired_goal']


        if actor_lower == 'random':
            a = env.action_space.sample()
        else:
            a = actor_lower(np.concatenate([o['observation'], sub_goal], axis=0))

        # Step the env
        o2, r, d, _ = env.step(a)


        if render:
            env.visualise_sub_goal(sub_goal, lower_achieved_whole_state=lower_achieved_whole_state)  # visualise yay!
            env.render(mode='human')

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        if return_episode:
            # we need to assign the desired goal as the subgoals, and potentially the achieved goal as the
            # actor position not the full state.
            # but tbh, achieved/desired goal should really be that of the entire state no? On the lower level
            # it must include actor position.
            o_store = o.copy()
            o2_store = o2.copy()
            o_store['desired_goal'] = sub_goal
            o2_store['desired_goal'] = sub_goal

            if lower_achieved_whole_state:
                o_store['achieved_goal'] = o_store['full_positional_state']
                o2_store['achieved_goal'] = o2_store['full_positional_state']

            else:
                o_store['achieved_goal'] = o_store['controllable_achieved_goal']
                o2_store['achieved_goal'] = o2_store['controllable_achieved_goal']
            r = env.compute_reward(o2_store['achieved_goal'], o2_store['desired_goal'], info = None)
            episode_lower.append([o_store, a, r, o2_store, d])  # add the full transition to the episode.
        # Super critical, easy to overlook step: make sure to update
        # most recent observation!

        o = o2
        # if either we've ended an episdoe, collected all the steps or have reached max ep len and
        # thus need to log ep reward and reset
        if d or (ep_len == int(max_ep_len)) or (t == int((n_steps - 1))):
            if return_episode:
                episode_buffer_lower.append(episode_lower)
                episode_lower = []
                episode_buffer_higher.append(episode_higher)
                episode_higher = []
            if summary_writer:
                with summary_writer.as_default():
                    if train:
                        print('Frame: ', t + current_total_steps, ' Return: ', ep_ret)
                        tf.summary.scalar('Episode_return', ep_ret, step=t + current_total_steps)
                    else:
                        print('Test Frame: ', t + current_total_steps, ' Return: ', ep_ret)
                        tf.summary.scalar('Test_Episode_return', ep_ret, step=t + current_total_steps)
            # reset the env if there are still steps to collect
            if t < n_steps - 1:
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # else it will be the end of the loop and of the function.
    if return_episode:
        return {'episodes_lower': episode_buffer_lower, 'episodes_higher': episode_buffer_higher, 'n_steps_lower': n_steps, 'n_steps_higher': higher_level_steps}
    return n_steps


# This is our training loop.
def training_loop(env_fn, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=500,
                  max_ep_len=300, save_freq=1, load=False, exp_name="Experiment_1", render=False, strategy='future',
                  num_cpus='max', use_higher_level = True, lower_achieved_whole_state = True):
    print('Begin')
    tf.random.set_seed(seed)
    np.random.seed(seed)
    test_env = env_fn()
    print('test_env-', test_env)
    num_cpus = psutil.cpu_count(logical=False)
    env = env_fn()
    # pybullet needs the GUI env to be reset first for our noncollision stuff to work.
    if render:
        print('Rendering Test Rollouts')
        test_env.render(mode='human')
        test_env.reset()


    # Little bit of short term conif
    use_higher_level = True
    replan_interval = 30
    lower_achieved_whole_state = True
    substitute_action = True


    # Get Env dimensions for networks
    obs_dim_higher = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    # now, our action can either be the  full state, or just the controllable aspects.
    if lower_achieved_whole_state:
        act_dim_higher = env.observation_space.spaces['full_positional_state'].shape[0]
    else:
        act_dim_higher = env.observation_space.spaces['controllable_achieved_goal'].shape[0]

    if use_higher_level:
        obs_dim_lower = env.observation_space.spaces['observation'].shape[0] + act_dim_higher
    else:
        obs_dim_lower = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]

    act_dim_lower = env.action_space.shape[0]

    # higher level model
    act_limit_higher = env.ENVIRONMENT_BOUNDS
    SAC_higher = SAC_model(act_limit_higher, obs_dim_higher, act_dim_higher, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load, exp_name+'_higher')
    act_limit_lower = env.action_space.high[0]
    SAC_lower = SAC_model(act_limit_lower, obs_dim_lower, act_dim_lower, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load, exp_name+'_lower')
    # Experience buffer
    replay_buffer_higher = HERReplayBuffer(env, obs_dim_higher, act_dim_higher, replay_size, n_sampled_goal=4,
                                    goal_selection_strategy=strategy)
    replay_buffer_lower = HERReplayBuffer(env, obs_dim_lower, act_dim_lower, replay_size, n_sampled_goal=4,
                                    goal_selection_strategy=strategy)
    # Logging
    start_time = time.time()
    train_log_dir = 'logs/' + exp_name + str(int(start_time))
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    def update_models(model, replay_buffer, steps, batch_size):
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size)
            LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.train_step(batch)

    def train(env, s_i, max_ep_len, SAC_lower, SAC_higher, summary_writer, steps_collected, exp_name, total_steps, replay_buffer_lower, replay_buffer_higher,
              batch_size, epoch_ticker, use_higher_level, substitute_action, replan_interval, lower_achieved_whole_state):

        episodes = rollout_trajectories_hierarchially(n_steps=max_ep_len, env=env, start_state=s_i, max_ep_len = max_ep_len, actor_lower = SAC_lower.actor.get_stochastic_action,
                                        actor_higher = SAC_higher.actor.get_stochastic_action, replan_interval = replan_interval,lower_achieved_whole_state = lower_achieved_whole_state,
                                        substitute_action = substitute_action, use_higher_level = use_higher_level, summary_writer = summary_writer, current_total_steps = steps_collected,
                                        exp_name = exp_name, return_episode = True)

        steps_collected += episodes['n_steps_lower']
        [replay_buffer_lower.store_hindsight_episode(e) for e in episodes['episodes_lower']]
        [replay_buffer_higher.store_hindsight_episode(e) for e in episodes['episodes_higher']]
        update_models(SAC_lower, replay_buffer_lower, steps=max_ep_len, batch_size=batch_size)
        update_models(SAC_higher, replay_buffer_higher, steps=episodes['n_steps_higher'], batch_size=batch_size) # gradient step it the same as the number of loweer steps because there are more.

        if steps_collected >= epoch_ticker:
            SAC_lower.save_weights()
            SAC_higher.save_weights()
            epoch_ticker += steps_per_epoch

        return steps_collected, epoch_ticker

    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0
    epoch_ticker = 0

    s_i, s_g = None, None

    if not load:
        # collect some initial random steps to initialise
        episodes = rollout_trajectories_hierarchially(n_steps=start_steps, env=env, start_state=s_i, max_ep_len=max_ep_len,
                                        actor_lower='random',
                                        actor_higher=SAC_higher.actor.get_stochastic_action, replan_interval=replan_interval,
                                        lower_achieved_whole_state=lower_achieved_whole_state,
                                        substitute_action=substitute_action, use_higher_level=use_higher_level, summary_writer=summary_writer,
                                        current_total_steps=steps_collected,
                                        exp_name=exp_name, return_episode=True)

        steps_collected += episodes['n_steps_lower']
        [replay_buffer_lower.store_hindsight_episode(e) for e in episodes['episodes_lower']]
        [replay_buffer_higher.store_hindsight_episode(e) for e in episodes['episodes_higher']]
        update_models(SAC_lower, replay_buffer_lower, steps=max_ep_len, batch_size=batch_size)
        update_models(SAC_higher, replay_buffer_higher, steps=episodes['n_steps_higher'], batch_size=batch_size)

    # now act with our actor, and alternately collect data, then train.
    print('Done Initialisation, begin training')

    while steps_collected < total_steps:
        try:
            steps_collected, epoch_ticker = train(env, s_i, max_ep_len, SAC_lower, SAC_higher, summary_writer, steps_collected, exp_name,
                                                  total_steps, replay_buffer_lower, replay_buffer_higher, batch_size, epoch_ticker, use_higher_level,
                                                  substitute_action, replan_interval, lower_achieved_whole_state)
        except KeyboardInterrupt:
            txt = input("\nWhat would you like to do: ")
            if txt == 'v':
                print('Visualising')

                rollout_trajectories_hierarchially(n_steps = max_ep_len, env = test_env, start_state = s_i, max_ep_len = max_ep_len,
                                actor_lower = SAC_lower.actor.get_deterministic_action,
                                actor_higher = SAC_higher.actor.get_deterministic_action, replan_interval = replan_interval,
                                lower_achieved_whole_state = lower_achieved_whole_state,
                                substitute_action = substitute_action, use_higher_level = use_higher_level, train=False, render=render, summary_writer = summary_writer,
                                current_total_steps = steps_collected,
                                exp_name = exp_name, return_episode = True)
            print('Returning to Training.')
            if txt == 'q':
                raise Exception

# TODO: Visualise subgoal, and add use higher  etc as params.
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
                        default=400)  # fetch reach learns amazingly if 50, but not if 200 -why?
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--strategy', type=str, default='future')
    parser.add_argument('--higher_level', type=str2bool, default=True)

    args = parser.parse_args()

    experiment_name = 'hierarchial2_' + args.env + '_Hidden_' + str(args.hid) + 'l_' + str(args.l)

    training_loop(lambda: gym.make(args.env),
                  ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  gamma=args.gamma, seed=args.seed, epochs=args.epochs, load=args.load, exp_name=experiment_name,
                  max_ep_len=args.max_ep_len, render=True, strategy=args.strategy)
