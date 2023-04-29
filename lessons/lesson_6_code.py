# Chapter 2 
from threading import current_thread
import matplotlib.pyplot as plt
import numpy as np; 
import random
np.random.seed(6)

class MultiArmedBandit():
    """
    A class that implements the N-armed testbed environment
    
    Attributes
    ----------
    	levers : int
    		number of levers, same as the possible actions in the environment
    	q_star : list
    		value of each action, i.e., mean reward obtained when the corresponding action is selected
    	sampling_variance: int
    		variance of the returned reward selecting an action, the mean is q_star[action]
    
    Methods
    -------
    	action( action )
    		return the reward obtained selecting the action 'action'; this value is obtained 
    		sampling from a distribution with mean=q_star[action] and varaince=self.sampling_variance
    """
    
    def __init__( self, levers ):
        self.levers = levers  
        self.sampling_variance = 1 
        self.q_star = list(np.random.normal(0,self.sampling_variance,levers))
        # self.q_star = [np.random.normal(0,self.sampling_variance) for _ in range(levers)] 
    		
    def action( self, action ):
        return np.random.normal(self.q_star[action],self.sampling_variance)


def banditAlgorithm( env, eps=0, maxiters=1000 ):
    """
    Implements the Simple Bandit Algorithm
    
    Args:
    	env: instance of the multi-armed bandit environmet
    	eps: random value for the eps-greedy policy (probability of random action)
    	maxiters: number of steps to perform in the environment
    	
    Returns:
    	avg_reward: list of the rewards obtained during the training,, averaged from the first step to the last
    	Q: the updated value function after the training
    """	
    
    levers = env.levers
    Q = np.array([0 for _ in range(levers) ], dtype=float)
    N = np.array([0 for _ in range(levers) ], dtype=int)
    ep_reward = []; avg_reward = []
    
    for _ in range(maxiters):

        random_action = int(np.random.rand(1)*env.levers)
        best_action = np.argmax(Q)    
        action_tuple= [random_action,best_action]
        current_action= random.choices(action_tuple,[eps,1-eps])[0]
                
        reward  = env.action(current_action)
        N[current_action]= N[current_action] + 1
        Q[current_action] = Q[current_action] + (1/N[current_action])*(reward-Q[current_action])

        ep_reward.append(reward)
        avg_reward.append( np.mean(ep_reward) )
    
    return avg_reward, Q


def main():
    print( "\n************************************************" )
    print( "*   Welcome to the sixth lesson of the RL-Lab!   *" )
    print( "*             (Multi-Armed Bandit)               *" )
    print( "**************************************************" )
    
    # Hyperparameters 
    n_armed = 10
    training_steps = 5000
    
    # Training phase
    print( f"\n{n_armed}-armed testbed with {training_steps} training_steps" )
    bandit = MultiArmedBandit( levers=n_armed )
    eps_00, q_00 = banditAlgorithm( bandit, eps=0.00, maxiters=training_steps ) 
    eps_01, q_01 = banditAlgorithm( bandit, eps=0.01, maxiters=training_steps ) 
    eps_10, q_10 = banditAlgorithm( bandit, eps=0.10, maxiters=training_steps )
    
    # Computing and plotting the reward of the last steps
    print( f"\n\tLast epsiodes reward (with eps=0   ):", np.mean(eps_00[-20:]) )
    print( f"\tLast epsiodes reward (with eps=0.01):", np.mean(eps_01[-20:]) )
    print( f"\tLast epsiodes reward (with eps=0.1 ):", np.mean(eps_10[-20:]) )
    
    # Computing and plotting the optimal action found
    # print( "\n\tThe real optimal action is: ", bandit.q_star.argmax() )
    print( f"\tThe optimal actions found are: {q_00.argmax()} (eps=0), {q_01.argmax()} (eps=0.01), and {q_10.argmax()} (eps=0.1)" )
    
    # Repeat the experiment for 2000 episodes
    eps_00_average, eps_01_average, eps_10_average = [], [], []
    for _ in range(50):
    	bandit = MultiArmedBandit( levers=n_armed )
    	eps_00_average.append( banditAlgorithm( bandit, eps=0.00, maxiters=training_steps )[0] )
    	eps_01_average.append( banditAlgorithm( bandit, eps=0.01, maxiters=training_steps )[0] )
    	eps_10_average.append( banditAlgorithm( bandit, eps=0.10, maxiters=training_steps )[0] )
    
    eps_00_average = np.average(np.array(eps_00_average), axis=0)
    eps_01_average = np.average(np.array(eps_01_average), axis=0)
    eps_10_average = np.average(np.array(eps_10_average), axis=0)
    
    
    # Plot the results
    print( "\n\tPlotting data..." )
    t = np.arange(0, training_steps)
    _, ax = plt.subplots()
    ax.plot(t, eps_00_average, label="eps: 0", linewidth=3)
    ax.plot(t, eps_01_average, label="eps: 0.01", linewidth=3)
    ax.plot(t, eps_10_average, label="eps: 0.1", linewidth=3)
    plt.xlabel( "steps", fontsize=16)
    plt.ylabel( "average reward", fontsize=16)
    ax.grid()
    plt.ylim([0, None])
    plt.legend()
    plt.show()
	

if __name__ == "__main__":
	main()
	
