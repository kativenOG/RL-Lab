import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): 
    # Initialize the neural network
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) #input layer + hidden layer #1
    for _ in range(1,nLayer):
        model.add(Dense(nNodes, activation="relu")) #hidden layer 
    model.add(Dense(nOutputs, activation=last_activation)) #output layer
    return model

def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ): 
    actor_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    critic_optimizer = tf.keras.optimizers.Adam( learning_rate=0.001 ) 
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    critic_memory_buffer = []
    for ep in range(episodes):
    
        partial_memory_buffer = []
        state = env.reset()[0] 
        state = state.reshape(-1,2)
        ep_reward = 0
        while True:
        
            # action = env.action_space.sample() 
            # La prossim azioend eve essere scelta il base alla policy predetta dall'actor
            n_action = 3 
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(env.action_space.n,p=distribution)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.reshape(-1,2)
            
            partial_memory_buffer.append( [state,action,reward,next_state,terminated])
            critic_memory_buffer.append( list([state,action,reward,next_state,terminated])) 
            ep_reward += reward
            
            if terminated or truncated:  break
            state = next_state
        
        # Perform the actual training every 'frequency' episodes
        memory_buffer.append(partial_memory_buffer)
        partial_memory_buffer = []
        if ep %frequency == 0 and ep!=0: 
            updateRule( actor_net,critic_net,critic_memory_buffer, memory_buffer, actor_optimizer, critic_optimizer) # critic_memory_buffer
            critic_memory_buffer = []
            memory_buffer = []
    
        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )
    
    #Close the enviornment and return the rewards list
    env.close()
    return rewards_list

def A2C( actor_net, critic_net,critic_memory_buffer, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99, observation_number=None ): 
    # UPDATE RULE CRITIC 
    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle( critic_memory_buffer )
        #TODO: Compute the target and the MSE between the current prediction and the expected advantage 
        for instance in critic_memory_buffer:
            state, action, reward, next_state, done = instance  
            done = 1 if done else 0
            target = reward + (1 - done)*gamma*critic_net(next_state).numpy()[0][0] 
            # CRITIC TAPE 
            with tf.GradientTape() as critic_tape:
                #MSE:
                predicted = critic_net(state)
                objective= tf.math.square(predicted - target)
                grad = critic_tape.gradient(objective, critic_net.trainable_variables)
                critic_optimizer.apply_gradients( zip(grad, critic_net.trainable_variables) )

    # ACTOR TAPE 
    with tf.GradientTape() as actor_tape:
        objectives = []

        for instance in memory_buffer:
            instance = np.array(instance)
            state,action,reward,next_state,dones = instance[:,0],instance[:,1],instance[:,2],instance[:,3],instance[:,4]
            # The objective function, notice that:
            # The REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory) multiplied by advantage
            objectives = []
            target = 0 
            for i in range(len(state)):
                appo = actor_net(state[i])
                log_probs = tf.math.log(appo[0][action[i]])
                adv_a = reward[i] + gamma * critic_net(next_state[i]).numpy().reshape(-1)
                adv_b = critic_net(state[i]).numpy().reshape(-1)
                target += log_probs * (adv_a[0] - adv_b[0])
            objectives.append(target)
                
        # Computing the final objective to optimize, is the average between all the considered trajectories
        objective = -tf.math.reduce_mean(objectives)
        grad = actor_tape.gradient(objective,actor_net.trainable_variables)
        actor_optimizer.apply_gradients( zip(grad, actor_net.trainable_variables) )


# TODO: implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment
    """
    def step(self, action):
        previous_observation = np.array(self.env.state, dtype=np.float32)
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Prof Method 1:
        if observation[1]==1: reward = -10  
        elif terminated: reward = 1000  
        elif(previous_observation[0] - observation[0])>0 and observation[1] == 0 : reward = 1 # Sto andando verso sinistra e spingendo verso sinistra 
        elif(previous_observation[0] - observation[0])<0 and observation[1] == 2 : reward = 1  # Sto andando verso destra e spingendo verso destra 
        return observation, reward, terminated, truncated, info

        # actual_reward = 0 
        # epsilon = 1e03
        # if terminated: actual_reward = 10000000
        # elif (action == 1): actual_reward = -1000000
        # elif (action==0):
        #     if sum(observation) <= -1.6: actual_reward = -10000
        #     actual_reward = -1/(1.6+observation[0]+ epsilon) -observation[1]*100
        # elif (action==2):
        #     if sum(observation) >= 0.5: actual_reward = 1000000
        #     actual_reward = -1/(0.5-observation[0]+ epsilon) + observation[1]*100

	

def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	_training_steps = 2000

	# Crete the environment and add the wrapper for the custom reward function
	gymnasium.envs.register(
		id='MountainCarMyVersion-v0',
		entry_point='gymnasium.envs.classic_control:MountainCarEnv',
		max_episode_steps=1000
	)
	env = gymnasium.make( "MountainCarMyVersion-v0" )
		
	# Create the networks and perform the actual training
	actor_net = createDNN( None, None, nLayer=None, nNodes=None, last_activation=None )
	critic_net = createDNN( None, None, nLayer=None, nNodes=None, last_activation=None )
	rewards_training, ep_lengths = training_loop( env, actor_net, critic_net, None, frequency=None, episodes=_training_steps  )

	# Save the trained neural network
	actor_net.save( "MountainCarActor.h5" )

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, ep_lengths, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "length", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	
