import numpy as np
from tensorboardX import SummaryWriter
import gym
import time
from pytorch_shared import *
import torch
import pybullet
import reach2D
import pointMass
from SAC import *
from hierarchy import *
from torch.optim import Adam
from common import *
from gym import wrappers
from datetime import datetime

# ok, what do we want to do?
# Q1 - DO we want to learn from one long thing, or multiple demos?
# lets say we want to multiple demos.
# then what are we doing? Sample random start and end, sample at intervals of replan_interval.
# desired goal for higher level is the achieved goal of the final state oberveed
# we want to create observation / action pairs of each set of obs separated by the replan interval
# action is either the whole state, or just the controllable section depending on the flag.

def sample_relay_batch(obs, ags, higher_level_acts, lower_level_acts, replan_interval, relative):
    # first thing we need to do is sample a couple of trajectories
    num_trajectories = 80
    traj_indexes = np.random.choice(obs.shape[0], num_trajectories)

    # then, in each trajectory take a random start and end
    max_start_index = int(obs.shape[1]*0.2) # the start index can be anywhere in the first 20% of steps.
    end_indices = np.arange(int(obs.shape[1]*0.8), obs.shape[1]) # the end indice can be anywhere in the last 20% of steps
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
        if relative:
            # take the relative higher level act from replan_time_steps ago
            high_acts = traj_higher_level_acts[replan_time_steps[1:]] - traj_higher_level_acts[replan_time_steps[:-1]]
        else:
            high_acts = traj_higher_level_acts[replan_time_steps[1:]]
        # we also want the desired goal of the higher level, which is just the last ag
        goal = traj_ags[-1]
        # now tile it out so we have one for each higher level obs to later concat along the last dimension
        high_desired_goals = np.tile(goal,[len(high_obs),1])
        high_in = np.concatenate([high_obs, high_desired_goals], axis=-1)
        # now, the lower level. We want each lower level ob to have desired goal of the corresponding next higher level act.
        # obs is still obs.
        # act is the baseline act.
        replan_time_steps  = replan_time_steps[1:] # take only the 1th on because we want to use the higher level acts as
        # our lower level goals.
        # sample a lower_level obsfrom somewhere within the replan time window of each higher level step.
        # clip at 0 because otherwise we will get negative indices
        low_obs_indexes = np.clip(replan_time_steps - np.random.choice(replan_interval, len(replan_time_steps)), 0, None)
        low_obs = traj_obs[low_obs_indexes]
        #  should be the replan time_steps not the last action.
        # confimed - it is because of the redic velocity ma boi travels at only in the pointmass task.
        #low_goals =  traj_higher_level_acts[[-1] * len(replan_time_steps)] #np.tile(goal,[len(low_obs),1]) #
        low_goals = high_acts #traj_higher_level_acts[replan_time_steps] # surely low goals should be high acts? Actually I guess it doesn't matter
        low_acts = traj_acts[low_obs_indexes]
        low_in = np.concatenate([low_obs, low_goals], axis = -1)


        # add to the arrays
        high_in_array.append(high_in)
        high_out_array.append(high_acts)
        low_in_array.append(low_in)
        low_out_array.append(low_acts)

    return torch.as_tensor(np.concatenate(high_in_array), dtype=torch.float32).cuda(), torch.as_tensor(np.concatenate(high_out_array), dtype=torch.float32).cuda(), torch.as_tensor(np.concatenate(low_in_array), dtype=torch.float32).cuda(), torch.as_tensor(np.concatenate(low_out_array), dtype=torch.float32).cuda()


def step(obs, ags, higher_level_acts, lower_level_acts, replan_interval, relative, steps, summary_writer, low_policy, high_policy):
    high_in, high_out, low_in, low_out = sample_relay_batch(obs, ags, higher_level_acts, lower_level_acts, replan_interval, relative)
    mu_low, _, distrib_low = low_policy(low_in)
    mu_high, _, distrib_high = high_policy(high_in)
    low_loss, high_loss = ((mu_low - low_out) ** 2).mean(), ((mu_high - high_out) ** 2).mean()
    #low_loss, high_loss = -distrib_low.log_prob(low_out).mean(), -distrib_high.log_prob(high_out).mean()
    summary_writer.add_scalar('relay_low_loss', low_loss, steps)
    summary_writer.add_scalar('relay_high_loss', high_loss,steps)
    return low_loss, high_loss

def train_step(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts, replan_interval, relative, low_optimizer, high_optimizer, steps, train_summary_writer, low_policy, high_policy):
    low_optimizer.zero_grad()
    high_optimizer.zero_grad()
    low_loss, high_loss = step(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts, replan_interval, relative, steps, train_summary_writer, low_policy, high_policy)
    low_loss.backward()
    low_optimizer.step() #TODO -----------
    high_loss.backward()
    high_optimizer.step()
    loss = low_loss + high_loss
    return loss

def find_supervised_loss_low(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts, replan_interval, relative, low_optimizer, high_optimizer, steps, train_summary_writer, low_policy, high_policy):
    low_optimizer.zero_grad()
    high_in, high_out, low_in, low_out = sample_relay_batch(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts,
                                                            replan_interval, relative)
    mu_low, _, distrib_low = low_policy(low_in)
    low_loss = ((mu_low - low_out) ** 2).mean()
    train_summary_writer.add_scalar('relay_low_loss', low_loss, steps)
    return low_loss

def find_supervised_loss_high(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts, replan_interval, relative, low_optimizer, high_optimizer, steps, train_summary_writer, low_policy, high_policy):
    high_optimizer.zero_grad()
    high_in, high_out, low_in, low_out = sample_relay_batch(train_obs, train_ags, train_higher_level_acts, train_lower_level_acts,
                                                            replan_interval, relative)
    mu_high, _, distrib_high = high_policy(high_in)
    high_loss = ((mu_high - high_out) ** 2).mean()
    train_summary_writer.add_scalar('relay_high_loss', high_loss, steps)
    return high_loss



def test_step(valid_obs, valid_ags, valid_higher_level_acts, valid_lower_level_acts, replan_interval, low_policy, high_policy, valid_summary_writer, steps, relative):
    low_loss, high_loss = step(valid_obs, valid_ags, valid_higher_level_acts, valid_lower_level_acts, replan_interval, relative, steps, valid_summary_writer, low_policy, high_policy)
    return low_loss, high_loss

def pretrain(train_kwargs, valid_kwargs, steps):
    train_kwargs['steps'] = steps
    train_step(**train_kwargs)
    if steps % 50 == 0:
        valid_kwargs['steps'] = steps
        l_low, l_high = test_step(**valid_kwargs)
        print('Test Loss: ', steps, ' Low: ', l_low, ' High: ', l_high)

    steps += 1
    return steps

def train(low_model, high_model, replay_buffer_lower, replay_buffer_higher, rollout_rl_kwargs, valid_kwargs, steps_collected,
          epoch_ticker, steps_per_epoch):
    # collect and store the trajectories
    #
    # if steps_collected < 5000:
    #     rollout_rl_kwargs['actor_lower'] = 'random'
    # else:
    #     rollout_rl_kwargs['actor_lower'] = SAC_lower.actor.get_stochastic_action

    rollout_rl_kwargs['current_total_steps'] = steps_collected
    episodes = rollout_trajectories_hierarchially(**rollout_rl_kwargs)
    [replay_buffer_lower.store_hindsight_episode(e) for e in episodes['episodes_lower']]
    [replay_buffer_higher.store_hindsight_episode(e) for e in episodes['episodes_higher']]

    # take consecutive supervised and unsupervised steps.

    for j in range(episodes['n_steps_lower']//50):
        valid_kwargs['steps'] =  steps_collected+j
        l_low, l_high = test_step(**valid_kwargs)
        print('Test Loss: ', steps_collected+j, ' Low: ', l_low.detach().cpu().numpy(), ' High: ', l_high.detach().cpu().numpy())
    steps_collected += episodes['n_steps_lower']

    if steps_collected >= epoch_ticker:
        low_model.save_weights()
        high_model.save_weights()
        epoch_ticker += steps_per_epoch

    return steps_collected, epoch_ticker





def relay_learning(filepath, env, test_env, exp_name, n_steps, architecture, batch_size =100, load=False, relative=False,
                   play=False, max_ep_len = 150, replay_size=int(1e6), steps_per_epoch=10000, lr=1e-3, seed=0, replan_interval = 10,
                   lower_achieved_state='full_positional_state', substitute_action= True, use_higher_level=True,
                   use_supervision_with_RL=True, supervision_weighting=1,  use_RL_on_higher_level=True, pretrain_steps=10000):
    # all data comes as [sequence, timesteps, dimension] so that when we are doing relay learning in the
    # trajectory we can't make mistakes about trajectory borders

    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.load(filepath)
    obs = data['obs']
    ags = data['achieved_goals']
    higher_level_acts = data[lower_achieved_state+'s']
    lower_level_acts = data['acts']
    test_env.render(mode='human')
    test_env.reset()


    train_length = int(0.8 * (len(obs)))
    # this is a pretty wasteful amount of copying - pending translation to pytorch dataloader.
    train_obs, train_ags, train_higher_level_acts, train_lower_level_acts  = obs[:train_length, :,:], ags[:train_length, :,:], higher_level_acts[:train_length, :,:], lower_level_acts[:train_length, :,:]
    valid_obs, valid_ags, valid_higher_level_acts, valid_lower_level_acts = obs[train_length:, :,:], ags[train_length:, :,:], higher_level_acts[train_length:, :,:], lower_level_acts[train_length:, :,:]



    # Get Env dimensions for networks
    act_dim_higher = env.observation_space.spaces[lower_achieved_state].shape[0]
    obs_dim_higher = env.observation_space.spaces['observation'].shape[0] + \
                     env.observation_space.spaces['desired_goal'].shape[0]
    obs_dim_lower = env.observation_space.spaces['observation'].shape[0] + act_dim_higher
    act_dim_lower = env.action_space.shape[0]

    # higher level model
    act_limit_higher = env.ENVIRONMENT_BOUNDS
    act_limit_lower =  env.action_space.high[0]

    start_time = datetime.now()
    train_log_dir, valid_log_dir  = 'logs/' + str(start_time) + 'train_' +exp_name+'_:' ,  'logs/' + str(start_time)+ 'valid_' +exp_name+'_:'
    train_summary_writer, valid_summary_writer = SummaryWriter(train_log_dir), SummaryWriter(valid_log_dir)

    high_model = SAC_model(act_limit_higher, obs_dim_higher, act_dim_higher, architecture, lr=lr,  load=load, exp_name = exp_name+'_higher')
    #low_model = SAC_model(act_limit_lower, obs_dim_lower, act_dim_lower, architecture, lr=lr, load=load, exp_name=exp_name.replace('relay', 'HER2'))
    low_model = SAC_model(act_limit_lower, obs_dim_lower, act_dim_lower, architecture, lr=lr, load=load, exp_name = exp_name+'_lower')
    #TODO -----------
    high_policy = high_model.ac.pi
    low_policy = low_model.ac.pi


    high_optimizer = high_model.pi_optimizer
    low_optimizer = low_model.pi_optimizer

    replay_buffer_higher = HERReplayBuffer(env, obs_dim_higher, act_dim_higher, replay_size, n_sampled_goal=4,
                                           goal_selection_strategy='final')
    replay_buffer_lower = HERReplayBuffer(env, obs_dim_lower, act_dim_lower, replay_size, n_sampled_goal=4,
                                          goal_selection_strategy='final')



    steps = 0
    epoch_ticker = 0

    train_kwargs = {'train_obs': train_obs, 'train_ags': train_ags, 'train_higher_level_acts':train_higher_level_acts,
                    'train_lower_level_acts': train_lower_level_acts, 'replan_interval': replan_interval, 'low_policy':low_policy,
                    'high_policy': high_policy, 'high_optimizer':high_optimizer, 'low_optimizer':low_optimizer,
                    'steps': steps, 'relative': relative, 'train_summary_writer': train_summary_writer}


    valid_kwargs = {'valid_obs': valid_obs, 'valid_ags': valid_ags, 'valid_higher_level_acts': valid_higher_level_acts,
                    'valid_lower_level_acts': valid_lower_level_acts, 'replan_interval': replan_interval,
                     'low_policy': low_policy, 'high_policy': high_policy, 'steps': steps, 'relative': relative, 'valid_summary_writer': valid_summary_writer}

    rollout_viz_kwargs = {'env': test_env, 'max_ep_len': max_ep_len,
                'actor_lower': low_model.ac.get_deterministic_action, 'actor_higher': high_model.ac.get_deterministic_action,
                'replan_interval': replan_interval, 'lower_achieved_state': lower_achieved_state, 'train': False,
                'use_higher_level': use_higher_level, 'summary_writer':valid_summary_writer,
                          'render':True, 'current_total_steps': steps, 'exp_name': exp_name, 'relative': relative}

    if use_supervision_with_RL:
        supervised_kwargs, supervised_func_low, supervised_func_high = train_kwargs, find_supervised_loss_low, find_supervised_loss_high
    else:
        supervised_kwargs, supervised_func_low, supervised_func_high = None, None, None

    rollout_rl_kwargs = {'n_steps':max_ep_len, 'env':env,
                       'max_ep_len':max_ep_len, 'actor_lower':low_model.actor.get_stochastic_action,
                    'actor_higher':high_model.actor.get_stochastic_action, 'replan_interval':replan_interval,
                    'lower_achieved_state':lower_achieved_state, 'exp_name':exp_name,'substitute_action':substitute_action,
                    'current_total_steps':steps, 'use_higher_level': use_higher_level, 'summary_writer':train_summary_writer,
                    'return_episode':True, 'sub_goal_testing_interval':3, 'relative': relative,
                    'sub_goal_tester':low_model.actor.get_deterministic_action, 'replay_buffer_low': replay_buffer_lower,
                    'replay_buffer_high': replay_buffer_higher, 'model_low':low_model, 'model_high':high_model, 'batch_size':batch_size,
                    'supervised_kwargs':supervised_kwargs, 'supervised_func_low': supervised_func_low, 'supervised_func_high': supervised_func_high
                         , 'supervision_weighting': supervision_weighting, 'use_RL_on_higher_level': use_RL_on_higher_level}


    if play:
        #test_env.activate_movable_goal()

        while(1):
            rollout_viz_kwargs['n_steps'] = max_ep_len
            rollout_viz_kwargs['current_total_steps'] += 1
            rollout_trajectories_hierarchially(**rollout_viz_kwargs)

    print('Done Initialisation, begin training')


    while steps < n_steps:
        try:
            if steps < pretrain_steps and not load:
                steps =  pretrain(train_kwargs, valid_kwargs, steps)
            else:
                steps, epoch_ticker = train(low_model, high_model,replay_buffer_lower, replay_buffer_higher, rollout_rl_kwargs, valid_kwargs, steps,epoch_ticker, steps_per_epoch)

        except KeyboardInterrupt:
            txt = input("\nWhat would you like to do: ")
            if txt.isnumeric():
                rollout_viz_kwargs['n_steps'] = max_ep_len * int(txt)
                rollout_viz_kwargs['render'] = True
                rollout_trajectories_hierarchially(**rollout_viz_kwargs)
            print('Returning to Training.')
            if txt == 'q':
                raise Exception
            if txt == 's':
                high_model.save_weights()
                low_model.save_weights()

    high_model.save_weights()
    low_model.save_weights()



# python relay_learning.py --filepath collected_data/10000HER2_pointMass-v0_Hidden_256l_2.npz
# python relay_learning.py --filepath collected_data/20000HER2_pointMassObject-v0_Hidden_256l_2.npz --env pointMassObject-v0

#python relay_learning.py --filepath  collected_data/125750HER2_pointMassObjectDuo-v0_Hidden_256l_2.npz --env pointMassObjectDuo-v0
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default="")
    parser.add_argument('--env', type=str, default='pointMass-v0')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--relative', type=str2bool, default=False)
    parser.add_argument('--use_higher_level', type=str2bool, default=True)
    parser.add_argument('--replan_interval', type=int, default=10)
    parser.add_argument('--lower_achieved_state', type=str, default='full_positional_state') # 'controllable_achieved_goal' #'achieved_goal'#
    parser.add_argument('--substitute_action', type=str2bool, default=True)
    parser.add_argument('--use_supervision_with_RL', type=str2bool, default=True)
    parser.add_argument('--play', type=str2bool, default=False)
    parser.add_argument('--supervision_weighting', type=int, default=1)
    parser.add_argument('--use_RL_on_higher_level', type=str2bool, default=True)
    parser.add_argument('--pretrain_steps', type=int, default=10000)
    parser.add_argument('--max_ep_len', type=int, default=150)

    args = parser.parse_args()
    if args.exp_name is None:
        exp_name = 'relay'+args.env+'-'+args.lower_achieved_state+'-'+str(args.replan_interval)
    else:
        exp_name = args.exp_name
#relayfullpos_x
#relaycontrollable_x
#relayachieved_1
    # pybullet needs the GUI env to be reset first for our noncollision stuff to work.
    print('Experiment Name: ', exp_name)
    save_file(__file__, exp_name, args)

    env = gym.make(args.env)
    test_env = gym.make(args.env)
    relay_learning(args.filepath, env, test_env, exp_name, args.n_steps, [args.hid]*args.l, batch_size = args.batch_size,
                   load =  args.load, relative = args.relative, play= args.play, replan_interval = args.replan_interval,
                   lower_achieved_state=args.lower_achieved_state, substitute_action= args.substitute_action, use_higher_level=args.use_higher_level,
                   use_supervision_with_RL=args.use_supervision_with_RL, supervision_weighting=args.supervision_weighting,
                   use_RL_on_higher_level=args.use_RL_on_higher_level, pretrain_steps = args.pretrain_steps, max_ep_len=args.max_ep_len)



# TODO - Implement the train as you go, and train for 500k and see where it gets to. Make sure each subgoal is a separate episode?