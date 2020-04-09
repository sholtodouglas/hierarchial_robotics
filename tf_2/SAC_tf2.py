import numpy as np
import tensorflow as tf
import gym
import time
import datetime
from tqdm import tqdm
from tensorflow.keras.layers import Dense#, Flatten, Conv2D,Bidirectional, LSTM, Dropout
from tensorflow.keras import Model
import os
print(tf.__version__)

import pybullet
import reach2D
import pointMass
import ur5_RL

import tensorflow_probability as tfp
from gym import wrappers
from common import *
from tensorboardX import SummaryWriter
tfd = tfp.distributions


#@title Building Blocks and Probability Func{ display-mode: "form" }
EPS = 1e-8

def mlp(hidden_sizes=[32,], activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    for layer_size in hidden_sizes[:-1]:
      model.add(Dense(layer_size, activation=activation))
    # Add the last layer with no activation
    model.add(Dense(hidden_sizes[-1], activation=output_activation))
    return model


def count_vars(model):
    return sum([np.prod(var.shape.as_list()) for var in model.trainable_variables])

def gaussian_likelihood(x, mu, log_std):
    
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
  
def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(input_tensor=tf.math.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


    #@title Policies { display-mode: "form" }

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class mlp_gaussian_policy(Model):

  def __init__(self, act_dim, act_limit, hidden_sizes=[400,300], activation = 'relu', output_activation=None):
    super(mlp_gaussian_policy, self).__init__()
    self.mlp = mlp(list(hidden_sizes), activation, activation)
    self.mu = Dense(act_dim, activation=output_activation)
    self.log_std = Dense(act_dim, activation='tanh',  name='log_std')
    self.act_limit = act_limit

  def call(self, inputs):
    x = self.mlp(inputs)
    mu = self.mu(x)
    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = self.log_std(x)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)

    pdf = tfd.Normal(loc=mu,scale=std)
    # pi = pdf.sample()
    # logp_pi = tf.reduce_sum(pdf.log_prob(pi))
    pi = mu + tf.random.normal(tf.shape(input=mu)) * std


    logp_pi = gaussian_likelihood(pi, mu, log_std)
    
    
    # I suppose just put this in here as the ops would overwrite - means theres less reuse but eh, won't kill us to have a slightly different policy func for each algo. 
    mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    
    # make sure actions are in correct range
    action_scale = self.act_limit
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi, std, pdf
  

  # have this function because it makes it simplifies the code in the trajectory collection part
  # lets us just pass some arbitrary 'actor function' that makes decisions.
  # This vastly simplifies interoperability with GAIL and the representation learning
  def get_deterministic_action(self,o):

    mu,_,_,_,_ = self.call(o.reshape(1,-1))
  
    return mu[0]

  def get_stochastic_action(self, o):
    _, pi, _, _, _ = self.call(o.reshape(1,-1))
    return  pi[0]
#@title Replay Buffer { display-mode: "form" }
# End Core, begin SAC.
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.acts_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

#@title SAC Model{ display-mode: "form" }
class SAC_model():
  
  def __init__(self, act_limit, obs_dim, act_dim, hidden_sizes,lr = 0.0001,gamma = None, alpha = None, polyak = None,  load = False, exp_name = 'Exp1', path = 'saved_models/', replay_buffer=None):
    self.act_limit = act_limit
    print(act_limit, '-----------------------')
    self.pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.gamma = gamma
    self.alpha = alpha
    self.polyak = polyak
    self.load = load
    self.exp_name = exp_name
    self.path = path
    self.create_networks(obs_dim, act_dim, hidden_sizes)

    
    
   
  def build_models(self,batch_size, obs_dim, act_dim):
    # run arbitrary data through the models to build them to the correct dimensions.
    self.actor(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    self.q_func1(tf.constant(np.zeros((batch_size,obs_dim+act_dim)).astype('float32')))
    self.q_func2(tf.constant(np.zeros((batch_size,obs_dim+act_dim)).astype('float32')))
    self.value(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    self.value_targ(tf.constant(np.zeros((batch_size,obs_dim)).astype('float32')))
    
  
  
    
  def create_networks(self,obs_dim, act_dim, hidden_sizes = [32,32], batch_size = 100, activation = 'relu'):
    
    # Get Env dimensions
 
    # Action limit for clamping: critically, assumes all dimensions share the same bound!

    
    self.actor = mlp_gaussian_policy(act_dim, self.act_limit, hidden_sizes, activation, None)
    # Create two q functions
    self.q_func1 = mlp(hidden_sizes+[1], activation, None)
    self.q_func2 = mlp(hidden_sizes+[1], activation, None)
    # create value and target value functions
    self.value = mlp(hidden_sizes+[1], activation, None)
    
    #collect them for saving/loading
    self.models = {'actor':self.actor, 'q1':self.q_func1, 'q2':self.q_func2, 'value':self.value}
    
    self.value_targ = mlp(hidden_sizes+[1], activation, None)
  
    #build the models by passing through arbitrary data.
    self.build_models(batch_size, obs_dim, act_dim)
    
    
    if self.load:
      
        self.load_weights(self.path)
      
      
    # Initializing targets to match main variables
    for v_main, v_targ in zip(self.value.trainable_variables, self.value_targ.trainable_variables):
      tf.compat.v1.assign(v_targ, v_main) 
    
    # Count variables
    var_counts = tuple(count_vars(model) for model in 
                       [self.actor, self.q_func1, self.q_func2, self.value])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d')%var_counts)
    
    
    
    
    
  @tf.function
  def update(self, batch):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as value_tape:
      x = batch['obs']
      x2 = batch['obs2']
      a = batch['act']
      r = batch['rew']
      d = batch['done']

      mu, pi, logp_pi, _, _ = self.actor(x)

      q1 = tf.squeeze(self.q_func1(tf.concat([x,a], axis=-1)))
      q1_pi = tf.squeeze(self.q_func1(tf.concat([x,pi], axis=-1)))
      q2 = tf.squeeze(self.q_func2(tf.concat([x,a], axis=-1)))
      q2_pi = tf.squeeze(self.q_func2(tf.concat([x,pi], axis=-1)))
      v = tf.squeeze(self.value(x))
      v_targ = tf.squeeze(self.value_targ(x2))

      # Min Double-Q:
      min_q_pi = tf.minimum(q1_pi, q2_pi)

      # Targets for Q and V regression
      q_backup = tf.stop_gradient(r + self.gamma*(1-d)*v_targ)
      v_backup = tf.stop_gradient(min_q_pi - self.alpha * logp_pi)

      # Soft actor-critic losses
      pi_loss = tf.reduce_mean(input_tensor=self.alpha * logp_pi - q1_pi)
      q1_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q1)**2)
      q2_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q2)**2)
      v_loss = 0.5 * tf.reduce_mean(input_tensor=(v_backup - v)**2)
      value_loss = q1_loss + q2_loss + v_loss

    # Policy train step
    # (has to be separate from value train step, because q1_pi appears in pi_loss)
    pi_gradients = actor_tape.gradient(pi_loss, self.actor.trainable_variables)
    self.pi_optimizer.apply_gradients(zip(pi_gradients, self.actor.trainable_variables))

    # Value train step
    value_variables = self.q_func1.trainable_variables + self.q_func2.trainable_variables + self.value.trainable_variables
    value_gradients = value_tape.gradient(value_loss, value_variables)
    #One notable byproduct of eager execution is that tf.control_dependencies() is no longer required, as all lines of code execute in order (within a tf.function, code with side effects execute in the order written).
    # Therefore, should no longer need controldependencies here. 
    self.value_optimizer.apply_gradients(zip(value_gradients, value_variables))

    # Polyak averaging for target variables
    for v_main, v_targ in zip(self.value.trainable_variables, self.value_targ.trainable_variables):
      tf.compat.v1.assign(v_targ, self.polyak*v_targ + (1-self.polyak)*v_main) 

    return pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi
  
  

  def get_weights(self):
      weights = []
      for name, model in self.models.items():
          weights.append([np.expand_dims(n.numpy(), axis = 0) for n in model.trainable_variables]) # must return the numpy in order for ray to work.
      return weights

  def set_weights(self, weights):
      # assign weights.
      for i, (name, model) in enumerate(self.models.items()):
          for distrib, master in zip(model.trainable_variables, weights[i]):
              tf.compat.v1.assign(distrib, np.squeeze(master, axis = 0))
      #update target to match
      for v_main, v_targ in zip(self.value.trainable_variables, self.value_targ.trainable_variables):
          tf.compat.v1.assign(v_targ, v_main)




  def load_weights(self, path = 'saved_models/'):
    try:
      print('Loading in network weights...')

      for name,model in self.models.items():
        model.load_weights(path+self.exp_name+'/'+name+'.h5')

      print('Loaded.')
    except:
        print("Failed to load weights.")



  def save_weights(self, path = 'saved_models/'):
      try:
          os.makedirs(path+self.exp_name)
      except Exception as e:
          #print(e)
          pass

      for name,model in self.models.items():
        model.save_weights(path+self.exp_name+'/'+name+'.h5')
      print("Model Saved at ", path+self.exp_name)

  # This is our training loop.


def SAC(env_fn, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=2000, epochs=100, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=2000,
                  max_ep_len=1000, save_freq=1, load=False, exp_name="Experiment_1", render=False):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    env, test_env = env_fn(), env_fn()
    test_env.render(mode='human')
    test_env.reset()

    env = wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    test_env = wrappers.FlattenDictWrapper(test_env, dict_keys=['observation', 'desired_goal'])
    # Get Env dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    SAC = SAC_model(act_limit, obs_dim, act_dim, ac_kwargs['hidden_sizes'], lr, gamma, alpha, polyak, load, exp_name)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Logging
    start_time = time.time()
    train_log_dir = 'logs/' + exp_name+str(int(start_time))
    summary_writer = SummaryWriter(train_log_dir)

    def update_models(model, replay_buffer, steps, batch_size):
        for j in range(steps):
            batch = replay_buffer.sample_batch(batch_size)
            LossPi, LossQ1, LossQ2, LossV, Q1Vals, Q2Vals, VVals, LogPi = model.update(batch)

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
    parser.add_argument('--env', type=str, default='pointMass-v0')
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






