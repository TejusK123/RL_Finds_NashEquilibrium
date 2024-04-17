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
import sys


"""
This one works !!
#possible continuation:
fiddle with rewards until both agents decide to cooperate instead of defect

figure out what is wrong with the twothirdsgame 





"""
class simpleparallelchoice(ParallelEnv):

	metadata = {"name" : "prisonersdilema", "render_modes" : ["human"], "game" : ["Prisoner", "Sexes", "Staghunt"]}
	def __init__(self, render_mode = None):
		self.memory = [0,0,0,0]
		self.possible_agents = ["player1", "player2"]


		self.render_mode = render_mode


	@functools.lru_cache(maxsize = None)
	def observation_space(self, agent):
		return(spaces.Discrete(2))


	@functools.lru_cache(maxsize = None)
	def action_space(self,agent):

		return(spaces.Discrete(2)) #either cooperate or defect



	def render(self):
		pass



	def reset(self, seed = None, options = None):

		self.agents = self.possible_agents[:]


		observations = {agent : False for agent in self.agents}


		infos = {agent : {} for agent in self.agents}


		return(observations, infos)



	def close(self):
		pass 


	def step(self, actions):

		
		if actions["player1"] == 1:
			self.memory[0] += 1
		elif actions["player1"] == 0:
			self.memory[1] += 1
		if actions["player2"] == 1:
			self.memory[2] += 1
		elif actions["player2"] == 0:
			self.memory[3] += 1

		print(self.memory)
		if not actions: 
			self.agents = []
			return({}, {}, {}, {}, {})


			# 1 == cooperate, 0 == defect
		if actions["player1"] == 1 and actions["player2"] == 1:
			rewards = {"player1" : 3, "player2" : 3}
		elif actions["player1"] == 1 and actions["player2"] == 0:
			rewards = {"player1" : 0, "player2" : 5}
		elif actions["player1"] == 0 and actions["player2"] == 1:
			rewards = {"player1" : 5, "player2" : 0}
		else:
			rewards = {"player1" : 1, "player2" : 1}

		print(rewards)
		truncations = {agent: True for agent in self.agents}

		terminations = {agent: True for agent in self.agents}

		infos = {agent : {} for agent in self.agents}


		observations = {"player1" : actions["player2"], "player2" : actions["player1"]}



		if truncations: 
			self.agents = []


		if self.render_mode == "human":
			self.render()


		return(observations, rewards, terminations, truncations, infos)


if __name__ == "__main__":
		
	ok = simpleparallelchoice()


	test.parallel_api_test(ok)


	# import numpy as np
	# import torch
	# import torch.nn as nn
	# import torch.optim as optim
	# from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
	# from torch.distributions.categorical import Categorical
	import supersuit as ss



	import stable_baselines3
	from stable_baselines3 import PPO
	from stable_baselines3.ppo import MlpPolicy

	if __name__ == "__main__":
		env = simpleparallelchoice()
		env = ss.pettingzoo_env_to_vec_env_v1(env)

		env = ss.concat_vec_envs_v1(env, 8, num_cpus = 0, base_class = "stable_baselines3")
		# print(env)

		model = PPO(MlpPolicy, env, verbose = 3, learning_rate = 1e-2, batch_size = 124, device = "cuda")
		print(model)

		model.learn(total_timesteps = 200000)
	
	






