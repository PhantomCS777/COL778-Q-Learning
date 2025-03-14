
from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np 
from typing import Tuple
import argparse
import torch
'''
import _ Agent
agent = Agent()
env = Env()
agent.train_policy(iterations = 1000)
 
'''

# NOTE
# what is tau
# should i use relu


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
                    discount_factor: float = 0.99,
                    tau=0.005,
                    iterations: int = 100000, 
                    eps_type: str = 'constant',
                    validation_runs: int = 100,
                    validate_every: int = 50000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 50000,
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

        self.eps_type = eps_type
        self.batch_size = batch_size
        self.lr = lr
        self.dqnet = DQNetwork()
        # what is tau
        self.tau = tau




    def choose_action(self, state, greedy = False):

        '''
        Right now returning random action but need to add
        your own logic
        '''
        #TO DO: You can add you code here
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
            
            obs = self.env.reset(i) #don't modify this
            
            #TO DO: You can add you code here

        return sum(rewards) / len(rewards), sum(dist) / len(dist)

    def visualize_policy(self, i: int) -> None:
        '''
        Args:
            i: total iterations done so for
        
        Create GIF visulizations of policy for visualization_runs
        '''
   
        for j in range(self.visualization_runs):
            obs = self.env.reset(j)  #don't modify this
            done = False
            images = [self.env.render()]

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
        #TO DO: You can add you code here

        

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations (integer).")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the input file.")
    args = parser.parse_args()
    #for part a
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    '''
    For part b:
        env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type='continious')
    '''
    env = HighwayEnv()
    qagent = DQNAgent(env,
                      iterations = 200000,
                      log_folder = args.output_folder)
    qagent.get_policy()