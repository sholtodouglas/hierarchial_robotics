# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
from SAC_pytorch import *
from HER import *
from common import *
import pointMass
from gym import wrappers


# used in expert collection, and with conversion of episodes to HER
# used in expert collection, and with conversion of episodes to HER
# extra info is for resetting determinsiticly.
def episode_to_trajectory(episode, goal_based = True):
	# episode arrives as a list of o, a, r, o2, d
	# trajectory is two lists, one of o s, one of a s.
	observations = []
	actions = []
	if goal_based:
		desired_goals = []
		achieved_goals = []
		controllable_achieved_goals = []
		full_positional_states = []
	for transition in episode:
		o, a, r, o2, d = transition
		if not goal_based:
			observations.append(o)
		else:
			observations.append(o['observation'])
		actions.append(a)
		if goal_based:
			desired_goals.append(o['desired_goal'])
			achieved_goals.append(o['achieved_goal'])
			controllable_achieved_goals.append(o['controllable_achieved_goal'])
			full_positional_states.append(o['full_positional_state'])


	return observations,actions,desired_goals, achieved_goals, controllable_achieved_goals, full_positional_states


def collect_expert(env, exp_name, n_steps, render, hierarchial, flatten, max_ep_len):


	print(render)
	if hierarchial:
		pass
	else:
		if flatten:
			env = wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
			obs_dim = env.observation_space.shape[0]
		else:
			obs_dim = env.observation_space.spaces['observation'].shape[0] + \
					  env.observation_space.spaces['desired_goal'].shape[0]


		act_dim = env.action_space.shape[0]
		act_limit = env.action_space.high[0]
		# Logging
		model = SAC_model(act_limit, obs_dim, act_dim,[256, 256],load=True, exp_name=exp_name)


		episodes = rollout_trajectories(n_steps=n_steps, env=env, max_ep_len=max_ep_len,
										goal_based=not flatten, actor=model.actor.get_deterministic_action, train=False,
										render=render, exp_name=exp_name, return_episode=True)

		action_buff = []
		observation_buff = []
		desired_goals_buff = []
		achieved_goals_buff = []
		controllable_achieved_goal_buff = []
		full_positional_state_buff = []
		for ep in episodes['episodes']:

			# quick fix for sub-optimal demos, just don't include as they are pretty rare
			# later, go collect more?
			if ep[-1][2] == 0: # i.e, if the reward of the last transition is not -1.

				observations,actions,desired_goals, achieved_goals, controllable_achieved_goals, full_positional_states= episode_to_trajectory(ep)
				action_buff.append(actions)
				observation_buff.append(observations)
				desired_goals_buff.append(desired_goals)
				achieved_goals_buff.append(achieved_goals)
				controllable_achieved_goal_buff.append(controllable_achieved_goals)
				full_positional_state_buff.append(full_positional_states)
			else:
				print('Rejecting Episode')


		np.savez('collected_data/'+ str(n_steps) + exp_name, acts=action_buff, obs=observation_buff, desired_goals=desired_goals_buff,
				 achieved_goals=achieved_goals_buff, controllable_achieved_goals= controllable_achieved_goal_buff,
				 full_positional_states=full_positional_state_buff)
		print('Saved at: \n collected_data/'+ str(n_steps) + exp_name+'.npz')



#python collect_expert.py --n_steps 10000 --render False

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='pointMass-v0')
	parser.add_argument('--exp_name', type=str, default=None)
	parser.add_argument('--n_steps', type=int, default=400)
	parser.add_argument('--render', type=str2bool, default=True)
	parser.add_argument('--hierarchial', type=str2bool,default=False)
	parser.add_argument('--hid', type=int, default=256)
	parser.add_argument('--l', type=int, default=2)
	parser.add_argument('--flatten', type=str2bool, default=False)
	parser.add_argument('--max_ep_len', type=int, default=100)


	args = parser.parse_args()
	if args.exp_name is None:
		exp_name = 'HER2_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)
	else:
		exp_name = args.exp_name
	env = gym.make(args.env)
	collect_expert(env, exp_name, args.n_steps, args.render, args.hierarchial, args.flatten, args.max_ep_len)

