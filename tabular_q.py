import re
from networkx import random_powerlaw_tree_sequence
from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np 
from typing import Tuple
import argparse

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
                    validation_runs: int = 100,
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
        self.avg_rewards_training = []
        self.avg_dist_training = []

    def get_action(self,state):
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
            from PIL import Image
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
            
            while(not done and k):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    qvalues = []
                    states = self.env.get_all_lane_states()
                    
                    #TO DO: You can add you code here

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

            while(not done and k):
                k += 1
                _ , _, done, _ = self.env.step(ignore_control_car = True)
                
                if(k % 20 == 0):

                    qvalues = []
                    states = self.env.get_all_speed_states()

                    #TO DO: You can add you code here
                    
    def get_policy(self):
        '''
        Learns the policy
        '''
        iterations = 0
        # How do i choose init
        state = env.reset()
        stop = False
        while(not stop):
            greedy = True
            if(self.eps_type == 'constant'):
                if(np.random.rand() < self.eps):
                    greedy = False
            
            # Non constant eps
            action = self.choose_action(state, greedy)

            new_state, reward, stop, _ = self.env.step(action)
            new_state = tuple(new_state) 
            if(state not in self.qtable):
                self.qtable[state] = np.zeros(5)
            if(new_state not in self.qtable):
                self.qtable[new_state] = np.zeros(5)
            
            if(stop):
                target = reward
                # How do i choose init
                if iterations%self.validate_every == 0:
                    print(f'Iteration: {iterations}')
                    cur_avg_reward,cur_avg_dist = self.validate_policy() 
                    self.avg_rewards_training.append(cur_avg_reward)
                    self.avg_dist_training.append(cur_avg_dist)
                iterations += 1
                state = self.env.reset()
                stop = False
                if(iterations == self.iterations):
                    stop = True
            else:
                target = reward + self.df * np.max(self.qtable[new_state])
            # self.qtable[state][action] += self.alpha * (target - self.qtable[state][action])
            self.qtable[state][action] = (1-self.alpha)*self.qtable[state][action] + self.alpha*target
            state = new_state

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()
    
    #For part a, b, c and d:
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    '''
    For part e and sub part a:
        env = get_highway_env(dist_obs_states = 5, reward_type = 'overtakes')
    For part e and sub part b:
        env = get_highway_env(dist_obs_states = 3, reward_type = 'dist')
    '''
    env = HighwayEnv()
    qagent = TabularQAgent(env, iterations = args.iterations,
                           log_folder = args.output_folder)
    qagent.get_policy()
    qagent.plot_avg_rewards_dist()
    qagent.visualize_policy(0) 