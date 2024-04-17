"""
Game trying to simulate is two thirds game

guess two thirds of whatever the average is from 
a number of agents choosing a number from 0-100


person who gets closest to two-thrids of average wins

#next try using CleanRL to train this model

"""


from pettingzoo.utils.env import ParallelEnv
import pettingzoo
import torch.nn as nn
import torch.optim as optim
import torch
from copy import copy
from gymnasium import spaces
from pettingzoo.utils import agent_selector
import pettingzoo.test as test
import functools
import math
import time



class CustomEnvironment(ParallelEnv):

    metadata = {"name" : "2/3rds game", "render_modes" : ["human"]}


    def __init__(self, render_mode = None, playeramt = 10):

        self.playeramt = playeramt


        self.possible_agents = [f"player{r}" for r in range(self.playeramt)]



        self.render_mode = render_mode

    @functools.lru_cache(maxsize = None)
    def observation_space(self, agent):
        #possible error
        #not really confident this is the right observation space
        return spaces.Discrete(101)

    @functools.lru_cache(maxsize = None)
    def action_space(self,agent):

        return spaces.Discrete(101)



    def render(self):
    #come up with some graphic that will come up
        pass

    def reset(self, seed = None, options = None):
        print("reset")
        self.agents = self.possible_agents[:]


        observations = {agent: False for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        return(observations, infos)

    def close(self):
        pass

        

    def step(self, actions):
        print("step")
        average = (sum(list((actions.values()))))//len(list((actions.values())))
        print(average)
        twothirds = average * (2/3)//1
        
        print(actions)
        if not actions:
            self.agents = []
            return({}, {}, {}, {}, {})

    #temporary think of a function that continuously gives rewards based on closeness to 2/3rds of average

    #update this reward function might be wack because it goes from 1 - veryhigh number: 
    #might want to minmaxscale it if the algorithm doesn't learn well

    
        #might be too close together the rewards
        #POSSIBLE FAILURE POINT
        rewards = {a: (1000/(abs(actions[a]-twothirds)+0.1)) for a in self.agents}

        print(rewards)
        truncations = {agent: True for agent in self.agents}

        terminations = {agent: True for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        averages = []


        #POSSIBLE FAILURE POINT
        #i.e make sure that each agent has an observation for every other agent
        observations = {a : average for a in self.agents}
        print(observations)


        if self.render_mode == "human":
            self.render()

        if truncations:
            self.agents = []


        return(observations, rewards, terminations, truncations, infos)



ok = CustomEnvironment("human")

test.parallel_api_test(ok)



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical




import supersuit as ss



import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

if __name__ == "__main__":
    env = CustomEnvironment()
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, 8, num_cpus = 0, base_class = "stable_baselines3")
    # print(env)

    model = PPO(MlpPolicy, env, verbose = 3, learning_rate = 1e-2, batch_size = 128, device = "cuda")
    print(model)

    model.learn(total_timesteps = 300000)

    