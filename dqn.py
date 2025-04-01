
from matplotlib.collections import QuadMesh
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
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
# NOTE
# should i use relu


# class DQNetwork(torch.nn.Module):
#     def __init__(self):
#         super(DQNetwork, self).__init__()
#         self.fc1 = torch.nn.Linear(6, 32)
#         self.fc2 = torch.nn.Linear(32, 32)
#         self.fc3 = torch.nn.Linear(32, 5)

#     def forward(self, x):
#         x = torch.nn.functional.relu(self.fc1(x))
#         x = torch.nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class DQNetwork(torch.nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(6, 1000)
        self.bn1 = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 400)
        self.bn2 = torch.nn.BatchNorm1d(400)
        self.fc3 = torch.nn.Linear(400, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
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
                    validation_runs: int = 1000,
                    validate_every: int = 1000,
                    visualize_runs: int = 10, 
                    visualize_every: int = 5000,
                    log_folder:str = './',
                    lr = 0.001,
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
        self.replay_buff = deque(maxlen=10000)
        # self.optim = torch.optim.SGD(self.dqnet.parameters(), lr=self.lr,momentum=0.95,weight_decay=1e-4, nesterov=True)
        self.base_optim = torch.optim.Adam
        self.optim = SAM(self.dqnet.parameters(), self.base_optim, lr=self.lr, rho=2.0, adaptive=True)
        self.loss_func = torch.nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim.base_optimizer, step_size=20000, gamma=0.8)
        self.avg_rewards_training = []
        self.avg_dist_training = []

   

    # def choose_action(self, state, greedy = False):

    #     '''
    #     Right now returning random action but need to add
    #     your own logic
    #     '''
    #     if(greedy):
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #         with torch.no_grad():
    #             q_values = self.dqnet(state_tensor)
    #         return int(torch.argmax(q_values).item())
    #     return np.random.randint(0, 5)
    def choose_action(self, state, greedy=False):
        if greedy:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Switch to eval mode to use running statistics instead of batch statistics
            self.dqnet.eval()
            with torch.no_grad():
                q_values = self.dqnet(state_tensor)
            self.dqnet.train()  # Switch back to train mode if needed
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
                    qvalues = self.dqnet(states_tensor)
                    qvalues = qvalues[:, 0]
                    qvalues = qvalues.detach().cpu().numpy()

                    req_image = self.env.render_lane_state_values(qvalues)
                    Image.fromarray(req_image).save(f"{self.log_folder}/lane_value/{i}_{j}_{k}.png")

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
                    qvalues = self.dqnet(states_tensor)
                    qvalues = qvalues[:, 0]
                    qvalues = qvalues.detach().cpu().numpy()

                    req_image = self.env.render_speed_state_values(qvalues)
                    Image.fromarray(req_image).save(f"{self.log_folder}/speed_value/{i}_{j}_{k}.png")
            

    def run_episode(self, state):
        done = False
        while(not done):
            greedy = True
            if(self.eps_type == 'constant'):
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
        for iteration in range(self.iterations):
            state = self.env.reset()
            self.run_episode(state)
            if iteration % 10000 == 0:
                # qagent.plot_avg_rewards_dist()
                torch.save(self.dqnet.state_dict(), f"{self.log_folder}/dqnet_{iteration}.pth")
                torch.save(self.target_net.state_dict(), f"{self.log_folder}/target_net_{iteration}.pth")
            if iteration%self.validate_every == 0:
                print(f'Iteration: {iteration}')
                # cur_avg_reward,cur_avg_dist = self.validate_policy() 
                # self.avg_rewards_training.append(cur_avg_reward)
                # self.avg_dist_training.append(cur_avg_dist)

            if len(self.replay_buff) >= self.batch_size:
                batch = random.sample(self.replay_buff, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)


                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)


                # q_values = self.dqnet(states).gather(1, actions)
                # with torch.no_grad():
                #     next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)


                # target = rewards + self.df * next_q_values * (1 - dones)
                # loss = self.loss_func(target, q_values)
                # self.optim.zero_grad()
                # loss.backward()
                # self.optim.step()
                # self.scheduler.step()
                self.optim.zero_grad()
                q_values = self.dqnet(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                target = rewards + self.df * next_q_values * (1 - dones)
                def closure():
                    q_values = self.dqnet(states).gather(1, actions)
                    loss = self.loss_func(target, q_values)
                    loss.backward()
                    return loss
                loss = self.loss_func(target, q_values)
                loss.backward()
                self.optim.step(closure)
                self.scheduler.step()

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
    qagent = DQNAgent(env,
                      iterations = args.iterations,
                      log_folder = args.output_folder)
    env = HighwayEnv()
    qagent.get_policy()
    # qagent.visualize_policy(0)
    qagent.validate_policy()
    for _ in range(2):
        cur_avg_reward,cur_avg_dist = qagent.validate_policy() 
        qagent.avg_rewards_training.append(cur_avg_reward)
        qagent.avg_dist_training.append(cur_avg_dist)
    qagent.plot_avg_rewards_dist()
    # qagent.visualize_speed_value(args.iterations)
    # qagent.visualize_lane_value(args.iterations)