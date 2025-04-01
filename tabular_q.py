import re
from networkx import random_powerlaw_tree_sequence
from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np 
from typing import Tuple
import argparse, os 
from PIL import Image
import matplotlib.pyplot as plt
'''
This the optional class tempelate for the Tabular Q agent. 
You are free to use your own class template or 
modify this class tempelate 
'''



# NOTE
# init state
# what does get_policy return
# why update q while sampling


class TabularQAgent:

    def __init__(self, 
                    env: HighwayEnv, 
                    alpha: float = 0.1, 
                    eps: float = 0.75, 
                    discount_factor: float = 0.99,
                    iterations: int = 100000, 
                    eps_type: str = 'constant',
                    validation_runs: int = 1000,
                    validate_every: int = 1000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 5000,
                    log_folder:str = './'
                    ):

        #TO DO: You can add you code here
        self.df = discount_factor
        self.alpha = alpha
        self.env = env
        self.iterations = iterations
        self.validation_runs = validation_runs
        self.validate_every = validate_every
        self.visualization_runs = visualize_runs
        self.visualization_every = visualize_every
        self.log_folder = log_folder

        self.eps = eps
        self.eps_type = eps_type
        self.qtable = dict() 
        self.exp_decay = 0.99
        self.linear_decay = 0.007
        # if os.path.exists(f'{self.log_folder}/qtable.npy'):
        #     self.qtable = np.load(f'{self.log_folder}/qtable.npy', allow_pickle=True).item() 
        self.avg_rewards_training = []
        self.avg_dist_training = []
        self.avg_eps_training = []

    def eps_step(self):
        if(self.eps_type == 'exponential'):
            self.eps = self.eps * self.exp_decay
        elif(self.eps_type == 'linear'):
            self.eps = max(0.0001,self.eps - self.linear_decay)
        
    def get_action(self,state):
        state  = tuple(state)
        if state in self.qtable:
            return np.argmax(self.qtable[state])
        else:
            return ACTION_NO_OP

    def choose_action(self, state, greedy = False):

        '''
        Right now returning random action but need to add
        your own logic
        '''
        if(greedy):
            if(state not in self.qtable):
                self.qtable[state] = np.zeros(5)
                return np.random.randint(0, 5)
            return np.argmax(self.qtable[state])
        else:
            return np.random.randint(0, 5)
    def save_qtable(self):
        '''
        Save the qtable to a file
        '''
        np.save(f'{self.log_folder}/qtable.npy', self.qtable)
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
            
            state = self.env.reset(i)
            discount = 1 
            cur_reward = 0 
            while(True):
                action = self.get_action(state)
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
                action = self.get_action(state)
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
            
            while(not done):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    qvalues = []
                    states = self.env.get_all_lane_states()
                    
                    #TO DO: You can add you code here

                    for state in states:
                        state = tuple(state)
                        if(state not in self.qtable):
                            self.qtable[state] = np.zeros(5)
                        qvalues.append(self.qtable[state][ACTION_NO_OP])
                    
                    req_image = self.env.render_lane_state_values(qvalues)
                    Image.fromarray(req_image).save(f"{self.log_folder}/lane_value/{i}{j}{k}.png")


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

            while(not done):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    qvalues = []
                    states = self.env.get_all_speed_states()

                    #TO DO: You can add you code here
                    for state in states:
                        state = tuple(state)
                        if(state not in self.qtable):
                            self.qtable[state] = np.zeros(5)
                        qvalues.append(self.qtable[state][ACTION_NO_OP])
                    
                    req_image = self.env.render_speed_state_values(qvalues)
                    Image.fromarray(req_image).save(f"{self.log_folder}/speed_value/{i}{j}{k}.png")
                    
    def run_episode(self, state):
        done = False
        state = tuple(state)
        while(not done):
            greedy = True
            if(self.eps_type == 'constant'):
                if(np.random.rand() < self.eps):
                    greedy = False    
            elif(self.eps_type == 'exponential'):
                if(np.random.rand() < self.eps):
                    greedy = False
            elif(self.eps_type == 'linear'):
                if(np.random.rand() < self.eps):
                    greedy = False
            
            action = self.choose_action(state, greedy)
            next_state, reward, done, _ = self.env.step(action)
            new_state = tuple(next_state) 

            if(state not in self.qtable):
                self.qtable[state] = np.zeros(5)
            if(new_state not in self.qtable):
                self.qtable[new_state] = np.zeros(5)
            

            if(done):
                target = reward
            else:
                target = reward + self.df * np.max(self.qtable[new_state])

            self.qtable[state][action] = (1 - self.alpha) * self.qtable[state][action] + self.alpha * target
            state = new_state


    def get_policy(self):
        '''
        Learns the policy
        '''
        for iteration in range(self.iterations):
            state = self.env.reset()
            self.run_episode(state)
            
            if iteration%self.validate_every == 0:
                print(f'Iteration: {iteration}')
                cur_avg_reward,cur_avg_dist = self.validate_policy() 
                self.avg_rewards_training.append(cur_avg_reward)
                self.avg_dist_training.append(cur_avg_dist)
                self.avg_eps_training.append(self.eps)
            if iteration%1000 == 0:
                self.eps_step()


    def plot_avg_rewards_dist(self,plot_id=None):
        """
        Plot the discounted return from start state and the maximum distance traveled from the start state by control
        """
        reward_path = f'{self.log_folder}/avg_rewards.png'
        dist_path = f'{self.log_folder}/avg_dist.png'
        if plot_id is not None:
            reward_path = f'{self.log_folder}/avg_rewards_{plot_id}.png'
            dist_path = f'{self.log_folder}/avg_dist_{plot_id}.png'
        
        plt.plot(self.avg_rewards_training)
        plt.xlabel('Iterations')
        plt.ylabel('Average discounted return')
        plt.title('Average discounted return vs Iterations')
        plt.savefig(reward_path)
        plt.close()

        plt.plot(self.avg_dist_training)
        plt.xlabel('Iterations')
        plt.ylabel('Average distance')
        plt.title('Average distance vs Iterations')
        plt.savefig(dist_path)
        plt.close()

    def plot_eps(self, plot_id=None):
        # plot running average of window 5 against iterations 

        window = 5
        running_avg = np.convolve(self.avg_rewards_training, np.ones(window)/window, mode='valid')
        reward_path = f'{self.log_folder}/mov_avg_reward.png'
        eps_path = f'{self.log_folder}/avg_eps.png'
        if plot_id is not None:
            eps_path = f'{self.log_folder}/avg_eps_{plot_id}.png'
            reward_path = f'{self.log_folder}/mov_avg_reward_{plot_id}.png'

        plt.plot(self.avg_eps_training, label='Epsilon')
        plt.xlabel('Iterations')
        plt.ylabel('Average epsilon')
        plt.title('Average epsilon vs Iterations')
        plt.savefig(eps_path)
        plt.close()

        plt.plot(running_avg, label='Running average')
        plt.xlabel('Iterations')
        plt.ylabel('Average discounted return')
        plt.title('Running average of discounted return vs Iterations')
        plt.savefig(reward_path)
        plt.close()
    def print_max_dist_reward(self,runs=100): 
        rewards = []
        dist = []
        
        for i in range(runs):
            state = self.env.reset(i)
            discount = 1 
            cur_reward = 0 
            while(True):
                action = self.get_action(state)
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
    parser.add_argument("--part", type=str, required=False, help="Part to run (a, b, c, d, e).")
    args = parser.parse_args()
    
    #For part a, b, c and d:
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    '''
    For part e and sub part a:
        env = get_highway_env(dist_obs_states = 5, reward_type = 'overtakes')
    For part e and sub part b:
        env = get_highway_env(dist_obs_states = 3, reward_type = 'dist')
    '''
    # env = HighwayEnv()
    default = True 
    try:
        if args.part != 'a' and args.part != None:
            default = False
    except:
        default = True
    
    if default:   
        qagent = TabularQAgent(env, iterations = args.iterations,
                            log_folder = args.output_folder)
        qagent.get_policy()
        qagent.plot_avg_rewards_dist()
        qagent.visualize_policy(0) 
        qagent.save_qtable()
        qagent.visualize_lane_value(0)
        qagent.visualize_speed_value(0)
        qagent.print_max_dist_reward()

    discount_factors = [0.8,0.9,0.99,0.999] 
    learning_rates = [0.1,0.3,0.5,0.05] 
    exploration_strats = ["constant","exponential","linear"]
    reward_types = ['dist','overtakes'] 
    quantization_lvl = [5,3,7] 
    
    try:
        # print(args.part)
        if args.part not in ['a', 'b', 'c', 'd', 'e_a', 'e_b']:
            raise ValueError("Invalid part. Choose from: a, b, c, d, e_a, e_b.")
    except:
        # print("No part specified. Exiting.")
        exit() 
    # Part b 
    if args.part == 'b':
        for df in discount_factors:
            qagent = TabularQAgent(env, iterations = args.iterations, 
                                discount_factor = df, log_folder = args.output_folder)
            qagent.get_policy()
            qagent.plot_avg_rewards_dist("part_b_df_" + str(df))
    
    # Part c
    if args.part == 'c':
        for lr in learning_rates:
            qagent = TabularQAgent(env, iterations = args.iterations, 
                                alpha = lr, log_folder = args.output_folder)
            qagent.get_policy()
            qagent.plot_avg_rewards_dist("part_c_lr_" + str(lr))
    
    # Part d
    if args.part == 'd':
        for eps in exploration_strats:
            qagent = TabularQAgent(env, iterations = args.iterations, 
                                eps_type = eps, log_folder = args.output_folder)
            qagent.get_policy()
            qagent.plot_avg_rewards_dist("part_d_eps_" + str(eps))
            qagent.plot_eps("part_d_eps_" + str(eps))
        
    # Part e_a 
    if args.part == 'e_a':
        for rt in reward_types:
            env = get_highway_env(dist_obs_states = 5, reward_type = rt)
            qagent = TabularQAgent(env, iterations = args.iterations, 
                                    log_folder = args.output_folder) 
            qagent.get_policy()
            qagent.plot_avg_rewards_dist("part_e_a_rt_" + str(rt))
    # Part e_b
    if args.part == 'e_b':
        for qt in quantization_lvl:
            env = get_highway_env(dist_obs_states = qt, reward_type = 'dist')
            qagent = TabularQAgent(env, iterations = args.iterations, 
                                    log_folder = args.output_folder) 
            qagent.get_policy()
            qagent.plot_avg_rewards_dist("part_e_b_qt_" + str(qt))