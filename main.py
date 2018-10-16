#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from system import ControlSystem
#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch

def main():
    experiment= 'control-system-v0' #specify environments here
    env=ControlSystem()

    steps= env.timestep_limit #steps per episode    
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
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

    
    for i in range(episodes):
        print ("==== Starting episode no:",i,"====")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            #rendering environmet (optional)            
            env.render()
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))

            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            if i % 100 == 0:
                print ("Action at step", t ," :",action,"\n")
            
            observation,reward=env.step(action,t)
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
                # print ('\n\n')
                break
    total_reward+=reward_per_episode            
    print ("Average reward per episode {}".format(total_reward / episodes)    )


if __name__ == '__main__':
    main()    