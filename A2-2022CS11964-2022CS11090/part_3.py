from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np
from typing import Tuple
import argparse
import torch
import copy
from collections import deque
import random
from PIL import Image


'''
You have to FOLLOW the given tempelate. 
In aut evaluation we will call

#to learn policy
agent = BestAgent()
agent.get_policy()

#to evaluate we will call
agent.choose_action(state)
'''

class DQNetwork(torch.nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(6, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 5)
        
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.dropout2 = torch.nn.Dropout(p=0.1)
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class BestAgent:

    def __init__(self, iterations = 500000) -> None:
        self.env = get_highway_env()
        self.iterations = iterations
        self.dqnet = DQNetwork()
        self.tau = 0.005
        self.target_net = copy.deepcopy(self.dqnet)
        self.replay_buff = deque(maxlen=100000)
        # self.optim = torch.optim.SGD(self.dqnet.parameters(), lr=3e-4, momentum=0.6)
        self.optim = torch.optim.Adam(self.dqnet.parameters(), lr=3e-4)
        self.loss_func = torch.nn.MSELoss()
        self.log_folder = "./"
        self.eps = 0.75
        self.df = 0.9
        self.batch_size = 256
        self.avg_rewards_training = []
        self.avg_dist_training = []


    def choose_action(self, state):
        '''
        This function should give your optimal 
        action according to policy
        '''

        self.dqnet.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.dqnet(state_tensor)
        self.dqnet.train()
        return int(torch.argmax(q_values).item())

    def exp_action(self, state, greedy = True):
        if(greedy):
            self.dqnet.eval()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.dqnet(state_tensor)
            self.dqnet.train()
            return int(torch.argmax(q_values).item())
        return np.random.randint(0, 5)


    def run_episode(self, state):
        done = False
        while(not done):
            greedy = True

            self.eps = max(0.3, self.eps*0.999)
            if(np.random.rand() < self.eps):
                greedy = False

            action = self.exp_action(state, greedy)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buff.append((state, action, reward, next_state, done))
            state = next_state



    def get_policy(self):
        '''
        Learns the policy
        '''
        for iteration in range(1, self.iterations + 1):
            state = self.env.reset()
            self.run_episode(state)
            # if iteration % 10000 == 0:
            #     torch.save(self.dqnet.state_dict(), f"{self.log_folder}/dqnet_{iteration}.pth")
            #     torch.save(self.target_net.state_dict(), f"{self.log_folder}/target_net_{iteration}.pth")
            # if iteration%1000 == 0:
            #     print(f'Iteration: {iteration}')
            #     cur_avg_reward,cur_avg_dist = self.validate_policy() 
            #     self.avg_rewards_training.append(cur_avg_reward)
            #     self.avg_dist_training.append(cur_avg_dist)
                
            if len(self.replay_buff) >= self.batch_size:
                batch = random.sample(self.replay_buff, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)


                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)


                q_values = self.dqnet(states).gather(1, actions)
                self.target_net.eval()
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                self.target_net.train()

                target = rewards + self.df * next_q_values * (1 - dones)
                loss = self.loss_func(target, q_values)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


                for target_param, param in zip(self.target_net.parameters(), self.dqnet.parameters()):
                    target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        # torch.save(self.dqnet.state_dict(), f"{self.log_folder}/dqnet_final.pth")
        # torch.save(self.target_net.state_dict(), f"{self.log_folder}/target_net_final.pth")

    def print_max_dist_reward(self,runs=100): 
        rewards = []
        dist = []
        
        for i in range(runs):
            state = self.env.reset(i)
            discount = 1 
            cur_reward = 0 
            while(True):
                action = self.exp_action(state, True)
                new_state, reward, stop, _ = self.env.step(action)
                new_state = tuple(new_state) 
                state = new_state 

                cur_reward += reward*discount 
                discount = discount*self.df
                if(stop):
                    dist.append(self.env.control_car.pos)
                    break 
            rewards.append(cur_reward)
        print(f'Maximum distance: {max(dist)}')
        print(f'Average Reward of Start State: {sum(rewards) / len(rewards)}')
    # def plot_avg_rewards_dist(self):
    #     """
    #     Plot the discounted return from start state and the maximum distance traveled from the start state by control
    #     """
    #     import matplotlib.pyplot as plt
    #     plt.plot(self.avg_rewards_training)
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Average discounted return')
    #     plt.title('Average discounted return vs Iterations')
    #     plt.savefig(f'{self.log_folder}/avg_rewards.png')
    #     plt.close()

    #     plt.plot(self.avg_dist_training)
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Average distance')
    #     plt.title('Average distance vs Iterations')
    #     plt.savefig(f'{self.log_folder}/avg_dist.png')
    #     plt.close()

    # def load_model(self, target_path, dqnet_path):
    #     self.target_net.load_state_dict(torch.load(target_path))
    #     self.dqnet.load_state_dict(torch.load(dqnet_path))


    # def validate_policy(self) -> Tuple[float, float]:
    #     '''
    #     Returns:
    #         tuple of (rewards, dist)
    #             rewards: average across validation run 
    #                     discounted returns obtained from first state
    #             dist: average across validation run 
    #                     distance covered by control car
    #     '''
    #     rewards = []
    #     dist = []

    #     for i in range(1000):
            
    #         state = self.env.reset(i) #don't modify this
    #         discount = 1 
    #         cur_reward = 0
    #         while(True):
    #             action = self.exp_action(state, True)
    #             new_state, reward, stop, _ = self.env.step(action)
    #             new_state = tuple(new_state) 
    #             state = new_state 

    #             cur_reward += reward*discount 
    #             discount = discount*self.df
    #             if(stop):
    #                 dist.append(self.env.control_car.pos)
    #                 break 
    #         rewards.append(cur_reward)

    #     return sum(rewards) / len(rewards), sum(dist) / len(dist)


    # def visualize_policy(self, i: int) -> None:
    #     '''
    #     Args:
    #         i: total iterations done so for
        
    #     Create GIF visulizations of policy for visualization_runs
    #     '''
   
    #     for j in range(10):
    #         state = self.env.reset(j)
    #         done = False
    #         images = [self.env.render()]
    #         #TO DO: You can add you code here
    #         while(True):
    #             action = self.exp_action(state, True)
    #             new_state, reward, stop, _ = self.env.step(action)
    #             new_state = tuple(new_state) 
    #             state = new_state 

    #             images.append(self.env.render()) 
    #             if(stop):
    #                 break 
    #         images = [Image.fromarray(i) for i in images]
    #         images[0].save(
    #             f"{self.log_folder}/output_{j}.gif",
    #             save_all=True,
    #             append_images=images[1:],
    #             duration=200,  # Time per frame in milliseconds
    #             loop=0,  # Loop forever
    #             optimize=True  # Optimize GIF for smaller file size
    #         )
    #         #TO DO: You can add you code here