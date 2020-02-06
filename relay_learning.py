import numpy as np
import tensorflow as tf
import gym
import time
import pybullet
import reach2D
import pointMass
from SAC import *
from hierarchy import *
from common import *
from gym import wrappers


# ok, what do we want to do?
# Q1 - DO we want to learn from one long thing, or multiple demos?
# lets say we want to multiple demos.
# then what are we doing? Sample random start and end, sample at intervals of replan_interval.
# desired goal for higher level is the achieved goal of the final state oberveed
# we want to create observation / action pairs of each set of obs separated by the replan interval
# action is either the whole state, or just the controllable section depending on the flag.

def sample_relay_batch(obs, ags, higher_level_acts, lower_level_acts, replan_interval = 15):
    # first thing we need to do is sample a couple of trajectories
    num_trajectories = 10
    traj_indexes = np.random.choice(obs.shape[0], num_trajectories)

    # then, in each trajectory take a random start and end
    max_start_index = int(obs.shape[1]*0.1) # the start index can be anywhere in the first 20% of steps.
    end_indices = np.arange(int(obs.shape[1]*0.9), obs.shape[1]) # the end indice can be anywhere in the last 20% of steps
    traj_start_indices = np.random.choice(max_start_index, num_trajectories)
    traj_end_indices = np.random.choice(end_indices, num_trajectories)

    # first things first, get it working using iterators - we can always optimise into matrix form later.
    high_in_array = []
    high_out_array = []
    low_in_array = []
    low_out_array = []


    for t in range(0,num_trajectories):
        traj_obs = obs[traj_indexes][t]
        traj_higher_level_acts = higher_level_acts[traj_indexes][t]
        traj_ags = ags[traj_indexes][t]
        traj_acts = lower_level_acts[traj_indexes][t]
        replan_time_steps = np.arange(traj_start_indices[t],traj_end_indices[t],replan_interval)
        # we want the higher level observations to be the obs at each higher level step
        # we want the higher level action to be the achieved_goal of the corresponding next higher level step
        # thats why we take 1: onwards for one, and up to the last one for the other.
        high_obs = traj_obs[replan_time_steps[:-1]]
        high_acts = traj_higher_level_acts[replan_time_steps[1:]]
        # we also want the desired goal of the higher level, which is just the last ag
        goal = traj_ags[-1]
        # now tile it out so we have one for each higher level obs to later concat along the last dimension
        high_desired_goals = np.tile(goal,[len(high_obs),1])
        high_in = np.concatenate([high_obs, high_desired_goals], axis=-1)
        # now, the lower level. We want each lower level ob to have desired goal of the corresponding next higher level act.
        # obs is still obs.
        # act is the baseline act.
        # sample a lower_level obsfrom somewhere within the replan time window of each higher level step.
        low_obs_indexes = replan_time_steps - np.random.choice(replan_interval, len(replan_time_steps))
        low_obs = traj_obs[low_obs_indexes]
        # TODO: this should really be the replan time_steps not the last action. Why is this so hard??
        low_goals =  traj_higher_level_acts[[-1] * len(replan_time_steps)] #np.tile(goal,[len(low_obs),1]) #
        low_acts = traj_acts[low_obs_indexes]
        low_in = np.concatenate([low_obs, low_goals], axis = -1)


        # add to the arrays
        high_in_array.append(high_in)
        high_out_array.append(high_acts)
        low_in_array.append(low_in)
        low_out_array.append(low_acts)

    return np.concatenate(high_in_array), np.concatenate(high_out_array), np.concatenate(low_in_array), np.concatenate(low_out_array)

def train_step(obs, ags, higher_level_acts, lower_level_acts, replan_interval, low_policy, high_policy ,optimizer, summary_writer, steps):
    high_in, high_out, low_in, low_out = sample_relay_batch(obs, ags, higher_level_acts, lower_level_acts, replan_interval)

    with tf.GradientTape() as tape:
        mu_low, _, _, _, _ = low_policy(low_in)
        mu_high,_,_,_,_ = high_policy(high_in)
        low_loss, high_loss = tf.reduce_sum(tf.losses.MSE(mu_low, low_out)), tf.reduce_sum(tf.losses.MSE(mu_high, high_out))
        BC_loss = low_loss + high_loss

    # I know this happens every time, but at the moment cbf building the model before this. Besides, isn't comp intensive.
    deterministic_variables = [x for x in low_policy.trainable_variables if 'log_std' not in x.name] + [x for x in high_policy.trainable_variables if 'log_std' not in x.name]
    BC_gradients = tape.gradient(BC_loss, deterministic_variables)
    optimizer.apply_gradients(zip(BC_gradients, deterministic_variables))

    with summary_writer.as_default():
        tf.summary.scalar('relay_low_loss', low_loss, step=steps)
        tf.summary.scalar('relay_high_loss', high_loss, step=steps)

    return BC_loss


def test_step(obs, ags, higher_level_acts, lower_level_acts, replan_interval, low_policy, high_policy, summary_writer, steps):
    high_in, high_out, low_in, low_out = sample_relay_batch(obs, ags, higher_level_acts, lower_level_acts, replan_interval)
    mu_low, _, _, _, _ = low_policy(low_in)
    mu_high, _, _, _, _ = high_policy(high_in)
    low_loss, high_loss = tf.reduce_sum(tf.losses.MSE(mu_low, low_out)), tf.reduce_sum(tf.losses.MSE(mu_high, high_out))
    with summary_writer.as_default():
        tf.summary.scalar('relay_low_loss', low_loss, step=steps)
        tf.summary.scalar('relay_high_loss', high_loss, step=steps)

    return low_loss, high_loss


def relay_learning(filepath, env, exp_name, n_steps, batch_size, goal_based, architecture, max_ep_len = 200):
    # all data comes as [sequence, timesteps, dimension] so that when we are doing relay learning in the
    # trajectory we can't make mistakes about trajectory borders

    replan_interval = 40
    lower_achieved_whole_state = True

    data = np.load(filepath)
    obs = data['obs']
    ags = data['achieved_goals']
    if lower_achieved_whole_state:
        higher_level_acts = data['full_positional_states']
        act_dim_higher = env.observation_space.spaces['full_positional_state'].shape[0]
    else:
        higher_level_acts = data['controllable_achieved_goals']
        act_dim_higher = env.observation_space.spaces['controllable_achieved_goal'].shape[0]
    lower_level_acts = data['acts']

    train_length = int(0.8 * (len(obs)))
    # this is a pretty wasteful amount of copying - pending translation to pytorch dataloader.
    train_obs, train_ags, train_higher_level_acts, train_lower_level_acts  = obs[:train_length, :,:], ags[:train_length, :,:], higher_level_acts[:train_length, :,:], lower_level_acts[:train_length, :,:]
    valid_obs, valid_ags, valid_higher_level_acts, valid_lower_level_acts = obs[train_length:, :,:], ags[train_length:, :,:], higher_level_acts[train_length:, :,:], lower_level_acts[train_length:, :,:]

    # Get Env dimensions for networks
    obs_dim_higher = env.observation_space.spaces['observation'].shape[0] + \
                     env.observation_space.spaces['desired_goal'].shape[0]
    obs_dim_lower = env.observation_space.spaces['observation'].shape[0] + act_dim_higher
    act_dim_lower = env.action_space.shape[0]

    # higher level model
    act_limit_higher = env.ENVIRONMENT_BOUNDS
    act_limit_lower =  env.action_space.high[0]

    start_time = time.time()
    train_log_dir, valid_log_dir  = 'logs/' + 'BC_train_' +exp_name+'_:' + str(start_time),  'logs/' + 'BC_valid_' +exp_name+'_:' + str(start_time)
    train_summary_writer, valid_summary_writer = tf.summary.create_file_writer(train_log_dir), tf.summary.create_file_writer(valid_log_dir)
    high_policy = mlp_gaussian_policy(act_dim=act_dim_higher, act_limit=act_limit_higher, hidden_sizes=architecture)
    low_policy = mlp_gaussian_policy(act_dim=act_dim_lower, act_limit=act_limit_lower, hidden_sizes=architecture)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    print('Done Initialisation, begin training')
    steps = 0
    while steps < n_steps:
        try:
            train_step(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts, replan_interval, low_policy, high_policy,
                       optimizer, train_summary_writer, steps)

            if steps % 500 == 0:
                l_low, l_high = test_step(valid_obs, valid_ags, valid_higher_level_acts, valid_lower_level_acts, replan_interval, low_policy,
                                          high_policy, valid_summary_writer, steps)
                print('Test Loss: ', steps,' Low: ', l_low, ' High: ', l_high)

            steps += 1

        except KeyboardInterrupt:
            txt = input("\nWhat would you like to do: ")
            if txt == 'v':
                print('Visualising')
                rollout_trajectories_hierarchially(n_steps=max_ep_len, env=env,
                                                   max_ep_len=max_ep_len,
                                                   actor_lower=low_policy.get_deterministic_action,
                                                   actor_higher=high_policy.get_deterministic_action,
                                                   replan_interval=replan_interval,
                                                   lower_achieved_whole_state=lower_achieved_whole_state,
                                                   train=False, render=True, use_higher_level = True,
                                                   current_total_steps=steps,
                                                   exp_name=exp_name)
            print('Returning to Training.')
            if txt == 'q':
                raise Exception



# python relay_learning.py --filepath collected_data/10000HER2_pointMass-v0_Hidden_256l_2.npz
# python relay_learning.py --filepath collected_data/20000HER2_pointMassObject-v0_Hidden_256l_2.npz --env pointMassObject-v0

if __name__ == '__main__':
    import argparse
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
        exp_name = 'relay_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)
    else:
        exp_name = args.exp_name
    env = gym.make(args.env)
    relay_learning(args.filepath, env, exp_name, args.n_steps, args.batch_size, args.goal_based, [args.hid]*args.l)