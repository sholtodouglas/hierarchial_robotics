# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
from SAC import *
from HER import *
from common import *
import pointMass
from gym import wrappers


# used in expert collection, and with conversion of episodes to HER
# used in expert collection, and with conversion of episodes to HER
# extra info is for resetting determinsiticly.

def rollout_trajectories_multi_block(n_steps, env, max_ep_len=200, actor=None, replay_buffer=None, summary_writer=None,
                         current_total_steps=0,
                         render=False, train=True, exp_name=None, z=None, s_g=None, return_episode=False,
                         replay_trajectory=None,
                         compare_states=None, start_state=None, goal_based=False, lstm_actor=None,
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
        episode_buffer = []
        episode = []

    if lstm_actor is not None:
        past_state = [None]

    for t in range(n_steps):

        obs_self = o['observation'][0:4]
        obs_block1 = o['observation'][4:8]
        obs_block2 = o['observation'][8:12]
        ag_block1 = o['observation'][4:6]
        ag_block2 = o['observation'][8:10]
        goal_block1 = o['desired_goal'][0:2]
        goal_block2 = o['desired_goal'][2:4]

        # if 1 is out of position focus on 1, if not go to block 2
        if env.compute_reward(ag_block1, goal_block1) < 0:
            obs = np.concatenate([obs_self, obs_block1])
            goal = goal_block1
        else:
            obs = np.concatenate([obs_self, obs_block2])
            goal = goal_block2

        a = actor(np.concatenate([obs, goal], axis=0))

        # Step the env
        o2, r, d, _ = env.step(a)

        if render:
            env.render(mode='human')

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer # dont use a replay buffer with HER, because we need to take entire episodes and do some computation so that happens after.
        if replay_buffer:
            replay_buffer.store(o, a, r, o2, d)

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
			obs_dim = 10#env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]


		act_dim = env.action_space.shape[0]
		act_limit = env.action_space.high[0]
		# Logging
		model = SAC_model(act_limit, obs_dim, act_dim,[256, 256],load=True, exp_name=exp_name.replace('Duo',''))


		episodes = rollout_trajectories_multi_block(n_steps=n_steps, env=env, max_ep_len=max_ep_len,
										goal_based=not flatten, actor=model.actor.get_deterministic_action, train=False,
										render=render, exp_name=exp_name, return_episode=True)

		action_buff = []
		observation_buff = []
		desired_goals_buff = []
		achieved_goals_buff = []
		controllable_achieved_goal_buff = []
		full_positional_state_buff = []
		successful = 0
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
				successful += 1
				print('Accepting Episode')
			else:
				print('Rejecting Episode')


		np.savez('collected_data/'+ str(successful*max_ep_len) + exp_name, acts=action_buff, obs=observation_buff, desired_goals=desired_goals_buff,
				 achieved_goals=achieved_goals_buff, controllable_achieved_goals= controllable_achieved_goal_buff,
				 full_positional_states=full_positional_state_buff)
		print('Saved at: \n collected_data/'+ str(successful*max_ep_len) + exp_name+'.npz')



#python collect_expert.py --n_steps 10000 --render False

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='pointMassObjectDuo-v0')
	parser.add_argument('--exp_name', type=str, default=None)
	parser.add_argument('--n_steps', type=int, default=100000)
	parser.add_argument('--render', type=str2bool, default=True)
	parser.add_argument('--hierarchial', type=str2bool,default=False)
	parser.add_argument('--hid', type=int, default=256)
	parser.add_argument('--l', type=int, default=2)
	parser.add_argument('--flatten', type=str2bool, default=False)
	parser.add_argument('--max_ep_len', type=int, default=2000)


	args = parser.parse_args()
	if args.exp_name is None:
		exp_name = 'HER2_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)
	else:
		exp_name = args.exp_name

	env = gym.make(args.env)
	env.activate_movable_goal()
	collect_expert(env, exp_name, args.n_steps, args.render, args.hierarchial, args.flatten, args.max_ep_len)

