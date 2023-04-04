import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
# Notice that the value function has only one output with a linear activation
# function in the last layer
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): 
     # Initialize the neural network
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) #input layer + hidden layer #1
    for _ in range(1,nLayer):
        model.add(Dense(nNodes, activation="relu")) #hidden layer 
    model.add(Dense(nOutputs, activation="linear")) #output layer
    
    return model

def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ): 
    actor_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    critic_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer,memory_buffer_partial = [],[]
    state = env.reset()[0]
    state = state.reshape(-1,4)
    
    ep_reward = 0
    for ep in range(episodes):
    
        state = env.reset()[0] 
        ep_reward = 0
        while True:
        
            action = env.action_space.sample() 
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.reshape(-1,4)
            
            memory_buffer.append( list([state,action,reward,next_state,terminated])) 
            ep_reward += reward
            
            if terminated or truncated:  break
            state = next_state
        
        #TODO: Perform the actual training every 'frequency' episodes
        memory_buffer.append(np.array(memory_buffer_partial)) # Cast to np Array for Slicing  
        memory_buffer_partial = []
        if ep %frequency == 0: 
            updateRule( actor_net,critic_net, memory_buffer, actor_optimizer, critic_optimizer)
            memory_buffer = []

        
        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )
    
    #Close the enviornment and return the rewards list
    env.close()
    return rewards_list

       

def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    ##Notes##
    One NN for The Value,state function and one for the policy prediction 
    

	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
	
	#TODO: implement the update rule for the critic (value function)
	for _ in range(10):
		# Shuffle the memory buffer
		np.random.shuffle( memory_buffer )
		#TODO: extract the information from the buffer
		# Tape for the critic
		with tf.GradientTape() as critic_tape:
			#TODO: Compute the target and the MSE between the current prediction
			# and the expected advantage 
			#TODO: Perform the actual gradient-descent process
			raise NotImplementedError

	#TODO: implement the update rule for the actor (policy function)
	#TODO: extract the information from the buffer for the policy update
	# Tape for the actor
	with tf.GradientTape() as actor_tape:
		#TODO: compute the log-prob of the current trajectory and 
		# the objective function, notice that:
		# the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
		# multiplied by advantage
		#TODO: compute the final objective to optimize, is the average between all the considered trajectories
		raise NotImplementedError
	

def main(): 
	print( "\n*************************************************" )
	print( "*  Welcome to the tenth lesson of the RL-Lab!   *" )
	print( "*                    (A2C)                      *" )
	print( "*************************************************\n" )

	_training_steps = 2500

	env = gymnasium.make( "CartPole-v1" )
	actor_net = createDNN( 4, 2, nLayer=2, nNodes=32, last_activation="softmax")
	critic_net = createDNN( 4, 1, nLayer=2, nNodes=32, last_activation="linear")
	rewards_naive = training_loop( env, actor_net, critic_net, A2C, episodes=_training_steps  )

	t = np.arange(0, _training_steps)
	plt.plot(t, rewards_naive, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	
