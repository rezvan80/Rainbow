# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

#import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
#from env import Env
from memory import ReplayMemory
from test import test


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
#parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=1, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)
def subtract_from_list(list_of_lists, value):
    for sublist in list_of_lists:
        value1 = value
        i = 0
        while value1 > 0 and i < len(sublist):
            if sublist[i] >= value1:
                sublist[i] -= value1
                value1 = 0
            else:
                value1 -= sublist[i]
                sublist[i] = 0
                i += 1
        # Now remove all zero values from this sublist
        # Use list comprehension for in-place update
        sublist[:] = [x for x in sublist if x != 0]
    return list_of_lists
import math
def fcc_encoder(arrs, ch_time , values):
    fcc=[]
    arrs=arrs.tolist()
    ch_time=ch_time.tolist()
    values=values.tolist()
    for arr, value in zip(arrs, values):
      sorted_arr = list(enumerate(arr))
      sorted_arr = [item for item in sorted_arr if (item[1] is not None and item[1] < value ) ]
      sorted_arr.sort(key=lambda item: item[1], reverse=True)
      indices_to_save=[]

      s=value
      for i in range(0 , len(sorted_arr)-1):

            diff=s-sorted_arr[i][1]
            s=sorted_arr[i][1]
            if diff>ch_time[sorted_arr[i][0]]:

              break

            indices_to_save.append(sorted_arr[i][0])

      fcc.append(
      sum([ch_time[i] for i in indices_to_save]) + arr[indices_to_save[-1]] - value
      if indices_to_save
      else 0
      )
    #fcc = nn.functional.softmax(torch.tensor(fcc).float()  , dim=0)
    return fcc
import osmnx as ox
import networkx as nx
import random
# Getting a street network for Manhattan, NYC
G = ox.graph_from_place("Piedmont, California, USA", network_type="drive")
# Download the street network within this bbox
G = ox.graph_from_bbox([ -122.2224914, 37.8222894,-122.2324914 , 37.8322894 ], network_type="drive")

#G = ox.graph_from_place("Oakland, California, USA", network_type="drive")
nodes = list(G.nodes())
reindex_mapping = {node: i for i, node in enumerate(nodes)}
G = nx.relabel_nodes(G, reindex_mapping)
nodes = list(G.nodes())
charging_station_nodes= random.sample(nodes, 3)

remaining_nodes = list(set(nodes) - set(charging_station_nodes))
start_node= random.sample(remaining_nodes, 20)
node_color = []

for node in G.nodes:
      if node in charging_station_nodes:  # Start node
          node_color.append('green')  # Start node in green
      elif node == start_node[0]:  # End node
          node_color.append('red')
      else:
          node_color.append('white')  # Other nodes in blue
#shortest_path = nx.shortest_path(G, source=start_node[0], target=charging_station_nodes[0], weight="length")
# Plot the graph and the shortest path
ox.plot_graph(G, node_color=node_color , node_size=30, figsize=(10, 8))
G = G.to_undirected()
from node2vec import Node2Vec

# Apply Node2Vec for graph embedding
node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=100, workers=4)

# Train the Node2Vec model
model = node2vec.fit(window=5, min_count=1)
import gymnasium as gym
from gymnasium  import spaces
from collections import deque
j=0
class charging_stationEnv5(gym.Env):
    def __init__(self, graph , charging_station_nodes , n_ev ):
        super(charging_stationEnv5, self).__init__()
        self.graph = graph
        self.j=0
        self.station_arr=np.array([[None]*n_ev ]*3)
        self.station_ch=list([[0] , [0] ,[0]])
        self.charging_station_nodes = [9 , 19 , 13]

        self.desierd_soc=list(np.random.uniform(0.6, 0.8,n_ev ))
        self.current_soc=list(np.random.uniform(0.4, 0.6,n_ev ))
        self.current_node= random.sample(list(set(list(self.graph.nodes())) - set(self.charging_station_nodes)),n_ev )

        self.current_soc=[0.51684445, 0.50067023, 0.53853516, 0.42557636, 0.46108417,
        0.44697881, 0.55068407, 0.43465217, 0.47451146, 0.5875858 ,
        0.5525329 , 0.43837829, 0.57131378, 0.47084906, 0.45499856,
        0.54569187, 0.53936948, 0.52403585, 0.47374796, 0.59744536]
        self.current_node=[7, 38, 33, 21, 0, 20, 25, 2, 12, 45, 11, 48, 15, 49, 40, 44, 52, 18, 27, 28]
         #self.path = [self.start_node]  # Initialize path tracker
        # Initialize path tracker
        self.charging_time=np.array([0]*n_ev)
        self.fcc= [deque([0]*30 , maxlen=30) for _ in range(n_ev )]
        self.label= [deque([0]*640 , maxlen=640) for _ in range(n_ev )]
        self.arrival_time=np.array([0]*3)
        self.num_envs=1
        self.test=False
        self.iteration= 0
        self.reward=[0]*n_ev
        self.state=[[0]*70]*n_ev
        self.done=[False]*n_ev
        self.average_reward=[0]
        # Define action and observation spaces
        self.action_space =spaces.Discrete(3)  # One action per node
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(70,), dtype=np.float32)
        self.travel_times=[0]*n_ev
        self.action=[0]*n_ev

        #self.average_reward=0
        self.average_distance=[0]
        self.distance=[0]*n_ev
        self.node=[0]*len(self.graph)
        #self.path=[[144], [290], [199], [193], [328], [302], [135], [274], [131], [35], [118], [29], [261], [104], [308], [207], [74], [86], [128], [211]]
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance[self.j]=0
        """ Reset the environment to the initial state (start node). """
        self.current_soc[self.j]=[0.51684445, 0.50067023, 0.53853516, 0.42557636, 0.46108417,
        0.44697881, 0.55068407, 0.43465217, 0.47451146, 0.5875858 ,
        0.5525329 , 0.43837829, 0.57131378, 0.47084906, 0.45499856,
        0.54569187, 0.53936948, 0.52403585, 0.47374796, 0.59744536][self.j]
        self.current_node[self.j]=[7, 38, 33, 21, 0, 20, 25, 2, 12, 45, 11, 48, 15, 49, 40, 44, 52, 18, 27, 28][self.j]
        #self.desierd_soc[self.j]=list(np.random.uniform(0.6, 0.8,1 ))[0]
        #self.current_soc[self.j]=list(np.random.uniform(0.4, 0.6,1 ))[0]
        #self.current_node[self.j]= random.sample(list(set(list(self.graph.nodes())) - set(self.charging_station_nodes)),1 )[0]

        #self.path = [self.start_node]  # Reset path tracker
        self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])],model.wv[str(self.charging_station_nodes[0])],model.wv[str(self.charging_station_nodes[1])],model.wv[str(self.charging_station_nodes[2])] ,np.array([model.wv[str(node_id)] for node_id in self.current_node ]).reshape(-1)) ,axis=0)
        #self.state[self.j]=np.array(charging_time(self.station_arr , self.charging_time))

        self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])],model.wv[str(self.charging_station_nodes[0])],model.wv[str(self.charging_station_nodes[1])],model.wv[str(self.charging_station_nodes[2])] ,np.array([model.wv[str(node_id)] for node_id in self.current_node ]).reshape(-1) ,np.array([self.desierd_soc[self.j]-self.current_soc[self.j]]), np.array(self.desierd_soc)-np.array(self.current_soc) ) ,axis=0)
        #self.state[self.j]=model.wv[str(self.current_node[self.j])]
        self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])],np.array([self.current_soc[self.j]-self.desierd_soc[self.j]]) , np.array(self.node))  , axis=0)
        #
        #self.state[self.j]=np.array(np.concatenate((model.wv[str(self.current_node[self.j])] ) , axis=0))
        #self.state[self.j]=np.array(self.fcc[self.j])
        #self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])] , np.where(self.station_arr == None, 0, self.station_arr).reshape(-1)) , axis=0)
        self.distance[self.j]=0
        return self.state[self.j] , {}

    def step(self , action ):


        self.done[self.j] = False
        shortest_path = nx.shortest_path(self.graph, source=self.current_node[self.j], target=self.charging_station_nodes[action], weight="length")
        """ Take a step in the environment with the given action. """

        self.station_arr[int(action)][int(self.j)]= nx.shortest_path_length(G, source=self.current_node[self.j], target=self.charging_station_nodes[action], weight="length")

        for row_idx in range(len(self.station_arr)):

               if row_idx != int(action):

                    self.station_arr[row_idx][self.j] = None

        distance=self.graph.get_edge_data(shortest_path[0] , shortest_path[1])[0]['length']
        self.travel_times[self.j] =distance
        self.current_soc[self.j]=self.current_soc[self.j]-distance*0.0001
        self.charging_time[self.j]=(self.desierd_soc[self.j]-self.current_soc[self.j])/0.002
        self.iteration += 1
        self.reward[self.j] = -distance

        self.distance[self.j]+=distance
        min_val = min(self.travel_times)
        self.station_ch=subtract_from_list(self.station_ch,min(self.travel_times)/10)
        self.label[self.j].extend(model.wv[str(self.current_node[self.j])])
        if self.node[self.current_node[self.j]] > 0:
           self.node[self.current_node[self.j]] -=1
        self.current_node[self.j] = shortest_path[1]
        self.node[self.current_node[self.j]] +=1
        #reward=1
        #done = self.current_node == self.goal_node
        # Plot the graph and the shortest path
        #self.plot_path()

        #if self.current_node[self.j] in self.charging_station_nodes:

        #if self.iteration == 4000:
              #self.station_ch=np.array([[0]]*3)
              #self.iteration= 0



        self.arrival_time = np.array([
              nx.shortest_path_length(
                    G, source=self.current_node[self.j], target=charging_station, weight="length"
                )
                for charging_station in self.charging_station_nodes
            ])

        self.fcc[self.j].extend(fcc_encoder(self.station_arr , self.charging_time , self.arrival_time))

              #reward=-np.sum(fcc_encoder(self.station_arr , self.charging_time , self.arrival_time))

        #if self.current_node[self.j] in self.charging_station_nodes:

        if self.current_node[self.j] in self.charging_station_nodes:
           self.average_reward.append(np.sum(self.station_ch[action]))
           ch=(self.desierd_soc[self.j]-self.current_soc[self.j])/0.001
           #self.reward[self.j]=10
           self.station_ch[action].append((self.desierd_soc[self.j]-self.current_soc[self.j])/0.002)
           self.reward[self.j]= 100*np.exp(-0.01*np.sum(self.station_ch[action]))
           self.reward[self.j]=100
           #self.reward[self.j]=10
           self.average_distance.append(self.distance[self.j])
           self.node[self.current_node[self.j]]-=1

           self.done[self.j] = True
           #action  = self.charging_station_nodes.index(self.current_node[self.j])  # Large reward for reaching the goal

        #self.reward[self.j]=100/(1+np.sum(self.station_ch[action]))
        #st_node[self.j])],model.wv[str(self.charging_station_nodes[0])],model.wv[str(self.charging_station_nodes[1])],model.wv[str(self.charging_station_nodes[2])] ,np.array([model.wv[str(node_id)] for node_id in self.current_node ]).reshape(-elf.reward[self.j]=-np.sum(charging_time(self.station_arr , self.charging_time ))/100
        #self.reward[self.j]=-np.sum(charging_time(self.station_arr , self.charging_time ))/100


        self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])],model.wv[str(self.charging_station_nodes[0])],model.wv[str(self.charging_station_nodes[1])],model.wv[str(self.charging_station_nodes[2])] ,np.array([model.wv[str(node_id)] for node_id in self.current_node ]).reshape(-1) , np.array([self.desierd_soc[self.j]-self.current_soc[self.j]]) , np.array(self.desierd_soc)-np.array(self.current_soc) ) ,axis=0)
        #self.state[self.j]=model.wv[str(self.current_node[self.j])]
        #self.state[self.j]=np.array(charging_time(self.station_arr , self.charging_time))
        self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])],np.array([self.desierd_soc[self.j]-self.current_soc[self.j]]) , np.array(self.node)) , axis=0)

        #self.state[self.j]=np.array(np.concatenate((model.wv[str(self.current_node[self.j])],np.array([self.current_soc[self.j]]),np.array([self.desierd_soc[self.j]])) , axis=0))
        #self.state[self.j]=np.concatenate((model.wv[str(self.current_node[self.j])] , np.where(self.station_arr == None, 0, self.station_arr).reshape(-1)) ,axis=0)

        return self.state[self.j] ,self.reward[self.j] ,self.done[self.j] ,  self.done[self.j] , {}


    def plot_path(self):
        """Visualizes the graph and highlights the visited path using ox.plot_graph."""
        # Add x and y coordinates as node attributes for OSMnx plotting

        # Highlight nodes in the path with distinct colors for start, end, and others
        node_color = []
        for node in self.graph.nodes:
            if node == self.current_node[self.j]:  # End node
                node_color.append('blue')
            elif node in self.charging_station_nodes:  # Start node
                node_color.append('green')  # Start node in green
            elif node in self.current_node:  # End node
                node_color.append('red')  # End node in red
            elif node == self.current_node[self.j]:  # End node
                node_color.append('blue')
            #elif node == self.start_node2:  # End node
                #node_color.append('red')  # End node in red

            else:
                node_color.append('white')  # Other nodes in blue

            # Plot the graph using ox.plot_graph
        ox.plot_graph(G, node_color=node_color , node_size=30, figsize=(10, 8))
n_ev=20
env=charging_stationEnv5(G, charging_station_nodes , n_ev)
avg_reward=[None]*n_ev
avg_Q=[None]*n_ev
state=[None]*n_ev
done=[None]*n_ev
next_state=[None]*n_ev
reward=[None]*n_ev
action_space = env.action_space.n
dqn = [Agent(args, env).to(device) for _ in range(20)]
# Agent


# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)
  mem = [load_memory(args.memory, args.disable_bzip_memory) for _ in range(20)]
else:
  mem = ReplayMemory(args, args.memory_capacity)
  mem = [ReplayMemory(args, args.memory_capacity) for _ in range(20)]
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
val_mem = [ReplayMemory(args, args.evaluation_size) for _ in range(20)]
T, done = 0, True
while T < args.evaluation_size:
  for i in range(n_ev):
    env.j=i
    if done[i]:
      state[i] , _ = env.reset()
      state[i] = torch.tensor(state[i], dtype=torch.float32, device='cpu')
    next_state[i], _, done[i] , _ , _ = env.step(np.random.randint(0, action_space))
    next_state[i] = torch.tensor(next_state[i], dtype=torch.float32, device='cpu')
    val_mem[i].append(state[i], -1, 0.0, done[i])
    state[i] = next_state[i]
    T += 1

if args.evaluate:
  for i in range(n_ev):
    
    dqn[i].eval()  # Set DQN (online network) to evaluation mode
    avg_reward[i], avg_Q[i] = test(args, 0, dqn[i], val_mem[i], metrics, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(np.men(avg_reward)) + ' | Avg. Q: ' + str(np.mean(avg_Q))
else:
  # Training loop
  for i in range(n_ev):
    dqn[i].train()
    done[i] = True
  for T in trange(1, args.T_max + 1):
    for i in range(n_ev):
      if done[i]:
        state[i] , _ = env.reset()
        state[i] = torch.tensor(state[i], dtype=torch.float32, device='cpu')
      if T % args.replay_frequency == 0:
        dqn[i].reset_noise()  # Draw a new set of noisy weights

        action[i] = dqn[i].act(state[i])  # Choose an action greedily (with noisy weights)
      next_state[i], reward[i], done[i] , _ , _ = env.step(action[i])  # Step
      next_state[i] = torch.tensor(next_state[i], dtype=torch.float32, device='cpu')
      if args.reward_clip > 0:
        reward[i] = max(min(reward[i], args.reward_clip), -args.reward_clip)  # Clip rewards
      mem[i].append(state[i], action[i], reward[i], done[i])  # Append transition to memory

    # Train and test
      if T >= args.learn_start:
        mem[i].priority_weight = min(mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        dqn[i].learn(mem[i])  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        dqn[i].eval()  # Set DQN (online network) to evaluation mode
        avg_reward[i], avg_Q[i] = test(args, T, dqn[i], val_mem[i], metrics, results_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(np.mean(avg_reward)) + ' | Avg. Q: ' + str(np.mean(avg_Q)))
        dqn[i].train()  # Set DQN (online network) back to training mode

        # If memory path provided, save it
        if args.memory is not None:
          save_memory(mem[i], args.memory, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        dqn[i].update_target_net()

      # Checkpoint the network
      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        dqn[i].save(results_dir, 'checkpoint.pth')

    state[i] = next_state[i]

env.close()
