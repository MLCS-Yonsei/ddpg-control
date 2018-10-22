#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from system import ControlSystem

import errno
import os
from datetime import datetime

from actor_net import ActorNet
from critic_net import CriticNet

import argparse

#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch

def main():
    enable_actuator_dynamics = True
    env=ControlSystem(enable_actuator_dynamics = enable_actuator_dynamics)

    steps= env.timestep_limit #steps per episode    
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)

    agent.load_model()

    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]    
    print ("Number of States:", num_states)
    print ("Number of Actions:", num_actions)
    print ("Number of Steps per episode:", steps)
    #saving reward:
    reward_st = np.array([0])

    log_dir = os.path.join(
        os.getcwd(), 'log',
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if enable_actuator_dynamics == True:
        filtered_log_dir = os.path.join(
            os.getcwd(), 'filtered_log',
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    y_hat_log_dir = os.path.join(
        os.getcwd(), 'y_hat_log',
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


    os.makedirs(log_dir)
    if enable_actuator_dynamics == True:
        os.makedirs(filtered_log_dir)
    os.makedirs(y_hat_log_dir)
    
    for i in range(episodes):
        print ("==== Starting episode no:",i,"====")
        observation = env.reset()
        reward_per_episode = 0
        actions_per_episode = []
        if enable_actuator_dynamics == True:
            filtered_action_per_episode = []

        for t in range(steps):
            #rendering environmet (optional)            
            env.render()
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))

            if action[0] > 1:
                action[0] = 1
            elif action[0] < 0:
                action[0] = 0

            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            actions_per_episode.append(action)
            # if i % 100 == 0:
            #     print ("Action at step", t ," :",action,"\n")
            
            if enable_actuator_dynamics == False:
                observation,reward,Y_plot,t_plot=env.step(action,t)
            elif enable_actuator_dynamics == True:
                observation,reward,filtered_action,Y_plot,t_plot=env.step(action,t)
                filtered_action_per_episode.append(filtered_action)
            
            # print ("Reward at step", t ," :",reward,"\n")
            #add y_t,y_t-1,action,reward,timestep to experience memory
            agent.add_experience(x,observation,action,reward,t)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (t == steps-1):
                print ('EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode)
                # print ("Printing reward to file")
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")

                # print("Y_plot")
                # plt.step(t_plot,Y_plot)
                # plt.grid()
                # plt.xlabel('t') 
                # plt.ylabel('y')
                # plt.show()

                # Save actions
                np.savetxt(log_dir + '/' + str(i).zfill(7) + '.txt', actions_per_episode)
                if enable_actuator_dynamics == True:
                    np.savetxt(filtered_log_dir + '/' + str(i).zfill(7) + '.txt', filtered_action_per_episode)
                np.savetxt(y_hat_log_dir + '/' + str(i).zfill(7) + '.txt', Y_plot)

                # save model
                if i % 100 == 0:
                    print('save')
                    # agent.save_model()
                # print ('\n\n')

                break
    total_reward+=reward_per_episode            
    print ("Average reward per episode {}".format(total_reward / episodes)    )


if __name__ == '__main__':
    main()    