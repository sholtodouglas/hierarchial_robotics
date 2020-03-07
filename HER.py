
TF = False
import gym
import pybullet
import pointMass #  the act of importing registers the env.
import ur5_RL
import time
from common import *
if TF:
    from SAC_tf2 import *
else:
    from SAC import *
from TD3 import *
import copy
import psutil
import numpy as np
import time
from pytorch_shared import *
import torch
from common import *
from tensorboardX import SummaryWriter
import multiprocessing as mp
from tqdm import tqdm
from natsort import natsorted, ns
from PrioritizedReplayBuffer import PER


#TODO Answer why reward scaling makes such a damn difference?

############################################################################################################
#Her with additional support for representation learning
############################################################################################################


# Agree with the stable baselines guys, HER is best implemented as a wrapper on the replay buffer.


# this is what we're working with at the moment.

# transitions arrive as -  obs, act, rew, next_obs, done
# but in HER, we need entire episodes.
# Ok, so one function for store episode, and that stores a bunch of transitions, either with strategy future or final.
# yeah so instead of storing transitions, store episdoes. Done? Then sample transitons the same way.
# how do we handle obs? take it as a dict at the ep stage, then when we're sampling for SAC .. flattened? Reward the same
# yeah. I reckon.

class HERReplayBuffer:
    """
    A simple FIFO experience replay buffer for HER agents.
    """

    def __init__(self, env, obs_dim, act_dim, size, n_sampled_goal = 4, goal_selection_strategy = 'future'):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.n_sampled_goal = n_sampled_goal
        self.env = env
        self.goal_selection_strategy = goal_selection_strategy
        self.ptr, self.size, self.max_size = 0, 0, size
        self.PER = PER(size)

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.PER.store(self.ptr) # store the index in the PER tree
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):
        #idxs = np.random.randint(0, self.size, size=batch_size)
        tree_idxs, idxs = self.PER.sample(batch_size)
        batch =  dict(obs=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.acts_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
        if TF:
            pass
        else:
            batch =  {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k, v in batch.items()}
        batch['PER_tree_idxs'] = tree_idxs
        return batch

    def update_priorities(self, tree_idxs, TD_errors):
        self.PER.batch_update(tree_idxs, TD_errors)

        # could be a goal, or both goal and z!
    def sample_achieved(self, transitions, transition_idx, strategy = 'future', encoder = None):
        if strategy == 'future':
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(transitions)))
            selected_transition = transitions[selected_idx]
            # here is where we get the obs and acts for the sequence up till there.
            # and here is where we will encode it, and get a nice z.
        elif strategy == 'final':
            selected_transition = transitions[-1]

        goal = selected_transition[3]['achieved_goal'] # TODO shouldn't this be of o2 not o? i.e index 3?
        return goal # and in future, z.


    # pass in an encoder if we desired reencoding of our representation learning trajectories.
    def store_hindsight_episode(self, episode, encoder = None):
        # ok, our episode comes in as a sequence of obs, acts. But we also want the actual rewards.
        # So really what we want is a sequence of transitions. Is that how we want to collect our episodes?

        # but we also want to be able to immediately convert our expert obs into our replay buffer?
        # so that they can be used immediately.
        # additionally, we want to be able to recompute z?

        # for the most part, when we store an obs we are storing o and d_g. # o, a, r, o2, d
        # when representation learning, we store o, z, d_g. from o,a,r,o2,d,z
        # remember, neural net inference is cheap, so maybe we can encode on the fly?


        for transition_idx, transition in enumerate(episode):


            o, a, r, o2, d = transition
            o = np.concatenate([o['observation'], o['desired_goal']])
            o2 = np.concatenate([o2['observation'], o2['desired_goal']])

            self.store(o, a, r, o2, d)

            if transition_idx == len(episode)-1:
                selection_strategy = 'final'
            else:
                selection_strategy = self.goal_selection_strategy

            sampled_achieved_goals = [self.sample_achieved(episode, transition_idx, selection_strategy) for _ in range(self.n_sampled_goal)]


            for goal in sampled_achieved_goals:


                o, a, r, o2, d = copy.deepcopy(transition)

                o['desired_goal'] = goal
                o2['desired_goal'] = goal

                r = self.env.compute_reward(goal, o2['desired_goal'], info = None) #i.e 1 for the most part

                o = np.concatenate([o['observation'], o['desired_goal']])
                o2 = np.concatenate([o2['observation'], o2['desired_goal']])

                self.store(o, a, r, o2, d)


# This is our training loop.
def training_loop(env_fn,  ac_kwargs=dict(), seed=0,
        steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=2000,
        max_ep_len=300, save_freq=1, load = False, exp_name = "Experiment_1", render = False, strategy = 'future', num_cpus = 'max'):

    print('Begin')

    #tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    print('Pretestenv')
    test_env = env_fn()
    print('test_env-',test_env)
    num_cpus = psutil.cpu_count(logical=False)
    env = env_fn()
    #pybullet needs the GUI env to be reset first for our noncollision stuff to work.
    if render:
        print('Rendering Test Rollouts')
        test_env.render(mode='human')
        test_env.reset()


    # Get Env dimensions
    obs_dim = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    replay_buffer = HERReplayBuffer(env, obs_dim, act_dim, replay_size, n_sampled_goal = 4, goal_selection_strategy = strategy)
    #Logging
    if TF:
        model = SAC_model(act_limit, obs_dim, act_dim, ac_kwargs['hidden_sizes'],lr, gamma, alpha, polyak,  load, exp_name)
    else:
        model = SAC_model(act_limit, obs_dim, act_dim, ac_kwargs['hidden_sizes'],lr, gamma, alpha, polyak,  load, exp_name, replay_buffer=replay_buffer)
        #model = TD3_model(act_limit, obs_dim, act_dim, ac_kwargs['hidden_sizes'], pi_lr=lr, q_lr=lr, gamma=gamma, alpha=alpha,polyak=polyak,load=load,exp_name=exp_name,replay_buffer=replay_buffer)
    # Experience buffer

    start_time = time.time()
    train_log_dir = 'logs/' + exp_name+str(int(start_time))
    summary_writer = SummaryWriter(train_log_dir)

    def update_models(model, replay_buffer, steps, batch_size):
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size)
            model.update(batch)

    def train(env,s_i, max_ep_len,model, summary_writer, steps_collected, exp_name, total_steps, replay_buffer, batch_size, epoch_ticker):

        episodes = rollout_trajectories(n_steps = max_ep_len,env = env,start_state=s_i, max_ep_len = max_ep_len, actor = model.actor.get_stochastic_action, summary_writer=summary_writer, current_total_steps = steps_collected, exp_name = exp_name, return_episode = True, goal_based = True)
        steps_collected += episodes['n_steps']
        [replay_buffer.store_hindsight_episode(e) for e in episodes['episodes']]
        update_models(model, replay_buffer, steps = max_ep_len, batch_size = batch_size)
        if steps_collected >= epoch_ticker:
            model.save_weights()
            epoch_ticker += steps_per_epoch

        return steps_collected, epoch_ticker

    # now collect epsiodes
    total_steps = steps_per_epoch * epochs
    steps_collected = 0
    epoch_ticker = 0

    s_i, s_g = None, None

    

    if not load:
    # collect some initial random steps to initialise

        episodes = rollout_trajectories(n_steps = start_steps,env = env, start_state=s_i,max_ep_len = max_ep_len, actor = 'random', summary_writer = summary_writer, exp_name = exp_name, return_episode = True, goal_based = True)
        steps_collected += episodes['n_steps']
        [replay_buffer.store_hindsight_episode(e) for e in episodes['episodes']]
        update_models(model, replay_buffer, steps = steps_collected, batch_size = batch_size)

    # now act with our actor, and alternately collect data, then train.
    print('Done Initialisation, begin training')

    while steps_collected < total_steps:
        try:
            steps_collected, epoch_ticker = train(env,s_i, max_ep_len,model, summary_writer, steps_collected, exp_name, total_steps, replay_buffer, batch_size, epoch_ticker)
        except KeyboardInterrupt:
            txt = input("\nWhat would you like to do: ")
            if txt == 'v':
                print('Visualising')
                rollout_trajectories(n_steps = max_ep_len,env = test_env, max_ep_len = max_ep_len, actor = model.actor.get_deterministic_action, summary_writer=summary_writer, current_total_steps = steps_collected, train = False, render = render, exp_name = exp_name, return_episode = True, goal_based = True)
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
    parser.add_argument('--max_ep_len', type=int, default=200) # fetch reach learns amazingly if 50, but not if 200 -why?
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--strategy', type=str, default='future')



    args = parser.parse_args()

    experiment_name = 'HER2_'+args.env+'_Hidden_'+str(args.hid)+'l_'+str(args.l)

    training_loop(lambda : gym.make(args.env),
      ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
      gamma=args.gamma, seed=args.seed, epochs=args.epochs, load = args.load, exp_name = experiment_name, max_ep_len = args.max_ep_len, render = True, strategy = args.strategy)
