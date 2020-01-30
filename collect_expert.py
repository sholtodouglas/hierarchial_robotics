import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
from SAC import *
from HER import *
from common import *
import pointMass
from gym import wrappers

flatten = False
ENV_NAME = 'pointMassObject-v0'#'reacher2D-v0'
#ENV_NAME = 'ur5_RL_relative-v0'
#ENV_NAME = 'ur5_RL-v0'
#ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'pointMass-v0'
env = gym.make(ENV_NAME)
#env.activate_roving_goal()
if flatten:
	env = wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
	obs_dim = env.observation_space.shape[0]
else:
	obs_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]

act_dim = env.action_space.shape[0]
#experiment_name = 'seg_reacher2D-v0_Hidden_128l_2'
#experiment_name = 'pos_cntrl_seg_reacher2D-v0_Hidden_128l_2'
#experiment_name = 'pos_cntrl_exp_pointMass-v0_Hidden_128l_2'
#experiment_name = 'no_reset_vel_pointMass-v0_Hidden_128l_2'
experiment_name = 'ultimate_pm_object'
experiment_name = 'HER2_pointMassObject-v0_Hidden_128l_2'
experiment_name  = 'HER2_linear_state_rep_sparse_reward_pointMass-v0_Hidden_128l_2'
extra_info = False
if experiment_name == 'HER2_pointMassObject-v0_Hidden_128l_2':
	extra_info = True
#experiment_name ='HER_ur5_RL_relative-v0_Hidden_128l_2'
#experiment_name ='HER_ur5_RL-v0_Hidden_128l_2'
#experiment_name = 'GAILpointMass-v0_Hidden_128l_2'
#experiment_name = 'GAILreacher2D-v0_Hidden_128l_2'
#env.activate_movable_goal()



SAC = SAC_model(env, obs_dim, act_dim, [128,128],load = True, exp_name = experiment_name)
n_steps = 10000
episodes = rollout_trajectories(end_on_reward = True, n_steps = n_steps,env = env, max_ep_len = 100,goal_based = not flatten, actor = SAC.actor.get_deterministic_action, train = False, render = True, exp_name = experiment_name, return_episode = True)

# episodes['episodes'] is a list of trajectories, of which each is an obs, ag dg, extrainfo in a single list for some reason.
#

np.savez('collected_data/obstacleboi2', episodes['episodes'])
action_buff = []
observation_buff = []
if extra_info:
	extra_info_buff = []
for ep in episodes['episodes']:
	if extra_info:
		observations, actions, info = episode_to_trajectory(ep, flattened=flatten, include_extra_info = True)

		extra_info_buff.append(info)
	else:
		observations, actions = episode_to_trajectory(ep, flattened = flatten)
	action_buff.append(actions)
	observation_buff.append((observations))
if extra_info:
	np.save('collected_data/'+str(n_steps)+experiment_name+'expert_extra_info',np.concatenate(extra_info_buff))
np.save('collected_data/'+str(n_steps)+experiment_name+'expert_actions',np.concatenate(action_buff))
np.save('collected_data/'+str(n_steps)+experiment_name+'expert_obs_',np.concatenate(observation_buff))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--flatten', type=bool, default=False)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--max_ep_len', type=int, default=400) # fetch reach learns amazingly if 50, but not if 200 -why?
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='future')


    args = parser.parse_args()

    experiment_name = 'HER2_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)

    training_loop(lambda : gym.make(args.env),
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = True, strategy = args.strategy)

