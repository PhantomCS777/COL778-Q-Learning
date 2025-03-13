from env import HighwayEnv, ACTION_NO_OP, get_highway_env
import numpy as np
'''
You have to FOLLOW the given tempelate. 
In aut evaluation we will call

#to learn policy
agent = BestAgent()
agent.get_policy()

#to evaluate we will call
agent.choose_action(state)
'''

class BestAgent:

    def __init__(self, iterations = 20000) -> None:
        self.env = get_highway_env()
        pass

    def choose_action(self, state):
        '''
        This function should give your optimal 
        action according to policy
        '''
        return np.random.randint(0,5)

    def get_policy(self) -> None: 
        '''
        This function should learn the policy
        '''
        pass