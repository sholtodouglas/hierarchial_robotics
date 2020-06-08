# hierarchial_robotics
The code for this [blog post](https://sholtodouglas.github.io/DoesHierarchialRLWorkYet/). 

This repository uses RL algorithms which are modularized versions of OpenAI's spinning up implementations to explore hierarchial RL. I found that they were excellent for extending to new algorithms. The code is split into models, rollout functions and replay buffers. Models expose their actor to the rollout function, which transfers episodes to the replay buffer for saving. Models then sample from the replay buffer. 

This made HER extremely easy, as it was just a replay buffer modificaiton. It also made hierarchy easy to implement, as it only required a new rollout function which used two models and two replay buffers. 

To run Soft Actor Critic on an environment, 

```python
python SAC.py --env []
```
To load saved weights, use --load True, which will load from the same --exp_name.

Other algorithms are HER (using SAC), TD3, hierarchy (based on Levy et al's Hierarchial Actor Critic). 

Once you have trained a model, you can use collect_expert.py to collect --n_steps on --env using the model loaded with --exp_name. 

You can then explore behavioural cloning with a command like 

```python
python BC.py --filepath collected_data/20000HER2_pointMassObject-v0_Hidden_256l_2.npz --env pointMassObject-v0
```

Similarly, relay learning is run like this 

```python
python relay_learning.py --filepath collected_data/20000HER2_pointMassObject-v0_Hidden_256l_2.npz --env pointMassObject-v0
```

