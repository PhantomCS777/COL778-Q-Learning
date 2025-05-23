
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
import _ Agent
agent = Agent()
env = Env()
agent.train_policy(iterations = 1000)
 
'''

class DQNetwork(torch.nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(6, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:

    def __init__(self, 
                    env: HighwayEnv, 
                    alpha: float = 0.1, 
                    eps: float = 0.75, 
                    discount_factor: float = 0.9,
                    tau=0.005,
                    iterations: int = 100000, 
                    eps_type: str = 'constant',
                    validation_runs: int = 1000,
                    validate_every: int = 1000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 5000,
                    log_folder:str = './',
                    lr = 0.0001,
                    batch_size = 256
                    ):

        self.env = env 
       
        self.eps = eps
        self.df = discount_factor
        self.alpha = alpha
        self.iterations = iterations
        self.validation_runs = validation_runs
        self.validate_every = validate_every
        self.visualization_runs = visualize_runs
        self.visualization_every = visualize_every

        self.log_folder = log_folder
        self.eps_type = eps_type
        self.batch_size = batch_size
        self.lr = lr
        self.dqnet = DQNetwork()
        self.tau = tau
        self.target_net = copy.deepcopy(self.dqnet)
        self.replay_buff = deque(maxlen=100000)
        # self.optim = torch.optim.SGD(self.dqnet.parameters(), lr=self.lr)
        self.optim = torch.optim.Adam(self.dqnet.parameters(), lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

        self.avg_rewards_training = []
        self.avg_dist_training = []


    def choose_action(self, state, greedy = False):

        '''
        Right now returning random action but need to add
        your own logic
        '''
        if(greedy):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.dqnet(state_tensor)
            return int(torch.argmax(q_values).item())
        return np.random.randint(0, 5)

    def validate_policy(self) -> Tuple[float, float]:
        '''
        Returns:
            tuple of (rewards, dist)
                rewards: average across validation run 
                        discounted returns obtained from first state
                dist: average across validation run 
                        distance covered by control car
        '''
        rewards = []
        dist = []

        for i in range(self.validation_runs):
            
            state = self.env.reset(i) #don't modify this
            discount = 1 
            cur_reward = 0
            while(True):
                action = self.choose_action(state, True)
                new_state, reward, stop, _ = self.env.step(action)
                new_state = tuple(new_state) 
                state = new_state 

                cur_reward += reward*discount 
                discount = discount*self.df
                if(stop):
                    dist.append(self.env.control_car.pos)
                    break 
            rewards.append(cur_reward)



        return sum(rewards) / len(rewards), sum(dist) / len(dist)

    def visualize_policy(self, i: int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create GIF visulizations of policy for visualization_runs
        '''
   
        for j in range(self.visualization_runs):
            state = self.env.reset(j)
            done = False
            images = [self.env.render()]
            #TO DO: You can add you code here
            while(True):
                action = self.choose_action(state, True)
                new_state, reward, stop, _ = self.env.step(action)
                new_state = tuple(new_state) 
                state = new_state 

                images.append(self.env.render()) 
                if(stop):
                    break 
            images = [Image.fromarray(i) for i in images]
            images[0].save(
                f"{self.log_folder}/output_{j}.gif",
                save_all=True,
                append_images=images[1:],
                duration=200,  # Time per frame in milliseconds
                loop=0,  # Loop forever
                optimize=True  # Optimize GIF for smaller file size
            )
            #TO DO: You can add you code here

    def visualize_lane_value(self, i:int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create image visulizations for no_op actions for particular lane
        '''
        
        for j in range(self.visualization_runs // 2):
            self.env.reset(j) #don't modify this
            done = False
            k = 0
            images = []
            while(not done):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):
                    states = self.env.get_all_lane_states()
                    
                    #TO DO: You can add you code here


                    states_tensor = torch.FloatTensor(states)
                    with torch.no_grad():
                        qvalues = self.dqnet(states_tensor)
                    qvalues = qvalues[:, ACTION_NO_OP]
                    qvalues = qvalues.detach().cpu().numpy()

                    req_image = self.env.render_lane_state_values(qvalues)
                    Image.fromarray(req_image).save(f"{self.log_folder}/{i}_{j}_{k}.png")

    def visualize_speed_value(self, i:int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create image visulizations for no_op actions for particular lane
        '''
        
        for j in range(self.visualization_runs // 2):
            self.env.reset(j) #don't modify this
            done = False
            k = 0
            images = []
            while(not done):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    states = self.env.get_all_speed_states()

                    #TO DO: You can add you code here

                    states_tensor = torch.FloatTensor(states)
                    with torch.no_grad():
                        qvalues = self.dqnet(states_tensor)
                    qvalues = qvalues[:, ACTION_NO_OP]
                    qvalues = qvalues.detach().cpu().numpy()

                    req_image = self.env.render_speed_state_values(qvalues)
                    Image.fromarray(req_image).save(f"{self.log_folder}/{i}_{j}_{k}.png")
            

    def run_episode(self, state):
        done = False
        while(not done):
            greedy = True
            if(self.eps_type == 'constant'):
                if(np.random.rand() < self.eps):
                    greedy = False
            else:
                self.eps = max(0.1, self.eps*0.99)
                if(np.random.rand() < self.eps):
                    greedy = False

            action = self.choose_action(state, greedy)
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
            if iteration % 10000 == 0:
                torch.save(self.dqnet.state_dict(), f"{self.log_folder}/dqnet_{iteration}.pth")
                torch.save(self.target_net.state_dict(), f"{self.log_folder}/target_net_{iteration}.pth")
            if iteration%self.validate_every == 0:
                print(f'Iteration: {iteration}')
                cur_avg_reward,cur_avg_dist = self.validate_policy() 
                self.avg_rewards_training.append(cur_avg_reward)
                self.avg_dist_training.append(cur_avg_dist)
                
            if len(self.replay_buff) >= self.batch_size:
                batch = random.sample(self.replay_buff, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)


                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)


                q_values = self.dqnet(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)


                target = rewards + self.df * next_q_values * (1 - dones)
                loss = self.loss_func(target, q_values)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


                for target_param, param in zip(self.target_net.parameters(), self.dqnet.parameters()):
                    target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)


    def plot_avg_rewards_dist(self):
        """
        Plot the discounted return from start state and the maximum distance traveled from the start state by control
        """
        import matplotlib.pyplot as plt
        plt.plot(self.avg_rewards_training)
        plt.xlabel('Iterations')
        plt.ylabel('Average discounted return')
        plt.title('Average discounted return vs Iterations')
        plt.savefig(f'{self.log_folder}/avg_rewards.png')
        plt.close()

        plt.plot(self.avg_dist_training)
        plt.xlabel('Iterations')
        plt.ylabel('Average distance')
        plt.title('Average distance vs Iterations')
        plt.savefig(f'{self.log_folder}/avg_dist.png')
        plt.close()

    def load_model(self, target_path, dqnet_path):
        self.target_net.load_state_dict(torch.load(target_path))
        self.dqnet.load_state_dict(torch.load(dqnet_path))
    
    def print_max_dist_reward(self,runs=100): 
        rewards = []
        dist = []
        
        for i in range(runs):
            state = self.env.reset(i)
            discount = 1 
            cur_reward = 0 
            while(True):
                action = self.choose_action(state,True)
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


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()
    # for part a
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    # For part b:
    
    # env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type='continious')
    
    qagent = DQNAgent(env,
                      iterations = args.iterations,
                      log_folder = args.output_folder)
    env = HighwayEnv()
    qagent.get_policy()
    qagent.visualize_policy(0)
    qagent.print_max_dist_reward()
    qagent.plot_avg_rewards_dist()
    # qagent.load_model("./dqn_gif/target_net_200000.pth", "./dqn_gif/dqnet_200000.pth")
    qagent.visualize_speed_value(args.iterations)
    qagent.visualize_lane_value(args.iterations)