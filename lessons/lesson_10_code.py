import warnings

from matplotlib import numpy; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections

def mse( network, dataset_input, target ):
    """
    Compute the MSE loss function
    """
    predicted_value = network( dataset_input )
    mse = tf.math.square(predicted_value - target)
    mse = tf.math.reduce_mean(mse) 
    # NB: still need reduce_mean to reduce the value to a scalar
    #     but keep it a tensor and perform BackProp
    return mse


# Implement the following functions as in the previous lessons
# Notice that the value function has only one output with a linear activation function in the last layer
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
    critic_memory_buffer= []
    for ep in range(episodes):
    
        partial_memory_buffer = []
        state = env.reset()[0] 
        state = state.reshape(-1,4)
        ep_reward = 0
        while True:
        
            # action = env.action_space.sample() 
            # La prossim azioend eve essere scelta il base alla policy predetta dall'actor
            n_action = 2 
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(env.action_space.n,p=distribution)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.reshape(-1,4)
            
            partial_memory_buffer.append( list([state,action,reward,next_state,terminated])) 
            critic_memory_buffer.append( list([state,action,reward,next_state,terminated])) 
            ep_reward += reward
            
            if terminated or truncated:  break
            state = next_state
        
        # Perform the actual training every 'frequency' episodes
        memory_buffer.append(partial_memory_buffer)
        partial_memory_buffer = []
        if ep %frequency == 0 and ep!=0: 
            updateRule( actor_net,critic_net, memory_buffer, critic_memory_buffer, actor_optimizer, critic_optimizer)
            critic_memory_buffer = []
            memory_buffer = []
    
        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )
    
    #Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def A2C( actor_net, critic_net, memory_buffer,critic_memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    ###Notes###
    One NN for The Value,state function and one for the policy prediction 
    
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
    # critic_memory_buffer = np.array(critic_memory_buffer)
    # memory_buffer = np.array(memory_buffer)
    # UPDATE RULE CRITIC 
    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle( critic_memory_buffer )
        #TODO: Compute the target and the MSE between the current prediction and the expected advantage 
        for instance in critic_memory_buffer:
            state, action, reward, next_state, done = instance  
            done = 1 if done else 0
            target = reward + (1 - done)*gamma*critic_net(next_state).numpy()[0][0] #done.astype(int)
            # CRITIC TAPE 
            with tf.GradientTape() as critic_tape:
                #MSE:
                # objective = mse(critic_net,state,target)
                predicted = critic_net(state)
                objective= tf.math.square(predicted - target)
                grad = critic_tape.gradient(objective, critic_net.trainable_variables )
                critic_optimizer.apply_gradients( zip(grad, critic_net.trainable_variables) )

    # ACTOR TAPE 
    #TODO: compute the log-prob of the current trajectory and 
    with tf.GradientTape() as actor_tape:
        objectives = []

        for instance in memory_buffer:
            instance = numpy.array(instance)
            state =      instance[:,0]
            action =     instance[:,1].tolist()
            reward =     instance[:,2].tolist()
            next_state = instance[:,3]

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

	
def main(): 
    print( "\n*************************************************" )
    print( "*  Welcome to the tenth lesson of the RL-Lab!   *" )
    print( "*                    (A2C)                      *" )
    print( "*************************************************\n" )
    
    _training_steps = 2500
    
    env = gymnasium.make( "CartPole-v1" )
    actor_net = createDNN( 4, 2, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN( 4, 1, nLayer=2, nNodes=32, last_activation="linear") # in uscita solo una dimensione 
    rewards_naive = training_loop( env, actor_net, critic_net, A2C, episodes=_training_steps  )
    
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_naive, label="A2C", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()	
