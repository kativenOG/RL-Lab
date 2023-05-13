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
    for ep in range(episodes):
    
        state = env.reset()[0] 
        state = state.reshape(-1,4)
        ep_reward,ep_lenght= 0,0
        while True:
        
            # La prossim azioend eve essere scelta il base alla policy predetta dall'actor
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(env.action_space.n,p=distribution)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.reshape(-1,4)
            memory_buffer.append([state,action,reward,next_state,terminated])
            ep_reward += reward
            ep_lenght+= 1
            if terminated or truncated:  break
            state = next_state
        
        # Perform the training every 'frequency'(10) episodes
        # memory_buffer.append(partial_memory_buffer)
        if ep %frequency == 0 and ep!=0: 
            updateRule( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer) # critic_memory_buffer
            memory_buffer = []
    
        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} len: {ep_lenght} (averaged: {np.mean(reward_queue):5.2f})" )
    
    #Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    ###Notes###
    One NN for The Value,state function and one for the policy prediction 
    
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
    # UPDATE RULE CRITIC 
    instance = numpy.array(memory_buffer)
    for _ in range(10):
        np.random.shuffle( instance) # Shuffle the memory buffer
        state = np.vstack(instance[:,0])
        reward = np.vstack(instance[:,2])
        next_state = np.vstack(instance[:,3])
        dones = np.vstack(instance[:,4])
        dones = np.vstack(dones) # BOHHH l'ha fatto il seba 

        target = reward + (1 - dones.astype(int))*gamma*critic_net(next_state).numpy()[0][0]
        # CRITIC TAPE 
        with tf.GradientTape() as critic_tape:
            predicted = critic_net(state)
            objective= tf.math.square(predicted - target)
            grad = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients( zip(grad, critic_net.trainable_variables) )

    # ACTOR TAPE 
    with tf.GradientTape() as actor_tape:
        objectives = [] # inutile
        objective = 0 
        actions  =  np.vstack(instance[:,1])
        probabilities= actor_net(state)
        probability = [x[actions[i][0]] for i,x in enumerate(probabilities)]
        log_probs = tf.math.log(probability)
        adv_a = reward + gamma * critic_net(next_state).numpy().reshape(-1)
        adv_b = critic_net(state).numpy().reshape(-1)
        objective += log_probs * (adv_a[0] - adv_b[0])
        objectives.append(objective)
                    
        # Computing the final objective to optimize, is the average between all the considered trajectories
        to_optimize = -tf.math.reduce_mean(objectives) # inutile 
        grad = actor_tape.gradient(to_optimize,actor_net.trainable_variables)
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
